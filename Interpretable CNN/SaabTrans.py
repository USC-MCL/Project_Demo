import numpy as np
from skimage.util.shape import view_as_windows

from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce

import matplotlib.pyplot as plt

import saab

class SaabTrans():
    def __init__(self, train_images, train_labels, kernel_sizes, num_kernels, energy_percent, use_num_images, use_classes):
        self.train_images=train_images
        self.train_labels=train_labels
        self.kernel_sizes=self.parse_list_string(kernel_sizes)
        if num_kernels:
    	    self.num_kernels=self.parse_list_string(num_kernels)
        else:
    	    self.num_kernels=None
        self.energy_percent=energy_percent
        self.use_num_images=use_num_images
        self.use_classes=use_classes
        self.images=train_images

    def parse_list_string(self,list_string):
	    """Convert the class string to list."""
	    elem_groups=list_string.split(",")
	    results=[]
	    for group in elem_groups:
		    term=group.split("-")
		    if len(term)==1:
			    results.append(int(term[0]))
		    else:
			    start=int(term[0])
			    end=int(term[1])
			    results+=range(start, end+1)
	    return results

    # convert responses to patches representation
    def window_process(self, samples, kernel_size, stride):#patches?
        '''
        Create patches
        :param samples: [num_samples, feature_height, feature_width, feature_channel]
        :param kernel_size: int i.e. patch size
        :param stride: int
        :return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

        '''
        n, h, w, c = samples.shape
        output_h = (h - kernel_size)//stride + 1
        output_w = (w - kernel_size)//stride + 1
        patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
        patches = patches.reshape(n, output_h, output_w, c*kernel_size*kernel_size)
        return patches


    def select_balanced_subset(self, images, labels, use_num_images, use_classes):
        '''
        select equal number of images from each classes
        '''
        # Shuffle
        num_total=images.shape[0]
        shuffle_idx=np.random.permutation(num_total)
        images=images[shuffle_idx]
        labels=labels[shuffle_idx]

        num_class=len(use_classes)
        num_per_class=int(use_num_images/num_class)
        selected_images=np.zeros((use_num_images,images.shape[1],images.shape[2],images.shape[3]))
        selected_labels=np.zeros(use_num_images)
        for i in range(num_class):
            images_in_class=images[labels==i]
            selected_images[i*num_per_class:(i+1)*num_per_class]=images_in_class[:num_per_class]
            selected_labels[i*num_per_class:(i+1)*num_per_class]=np.ones((num_per_class))*i

        # Shuffle again
        shuffle_idx=np.random.permutation(num_per_class*num_class)
        selected_images=selected_images[shuffle_idx]
        selected_labels=selected_labels[shuffle_idx]
        # For test
        # print(selected_images.shape)
        # print(selected_labels[0:10])
        # plt.figure()
        # for i in range (10):
        # 	img=selected_images[i,:,:,0]
        # 	plt.imshow(img)
        # 	plt.show()
        return selected_images,selected_labels


    def multi_Saab_transform(self):
        '''
        Do the PCA "training".
        :param images: [num_images, height, width, channel]
        :param labels: [num_images]
        :param kernel_sizes: list, kernel size for each stage,
            the length defines how many stages conducted
        :param num_kernels: list the number of kernels for each stage,
            the length should be equal to kernel_sizes.
        :param energy_percent: the energy percent to be kept in all PCA stages.
            if num_kernels is set, energy_percent will be ignored.
        :param use_num_images: use a subset of train images
        :param use_classes: the classes of train images
        return: pca_params: PCA kernels and mean
        '''

        num_total_images=self.images.shape[0]
        if self.use_num_images<num_total_images and self.use_num_images>0:
            sample_images, selected_labels=select_balanced_subset(self.images, self.labels, self.use_num_images, self.use_classes)
        else:
            sample_images=self.images
        # sample_images=images
        num_samples=sample_images.shape[0]
        num_layers=len(self.kernel_sizes)
        pca_params={}
        pca_params['num_layers']=num_layers
        pca_params['kernel_size']=self.kernel_sizes

        for i in range(num_layers):
            print('--------stage %d --------'%i)
            # Create patches
            # sample_patches=window_process(sample_images,kernel_sizes[i],kernel_sizes[i]) # nonoverlapping
            sample_patches=self.window_process(sample_images,self.kernel_sizes[i],1) # overlapping
            h=sample_patches.shape[1]
            w=sample_patches.shape[2]
            # Flatten
            sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

            #sample_patches 
            if not self.num_kernels is None:
                num_kernel=self.num_kernels[i]
            
            saab0=saab.Saab(num_kernels=num_kernel)
            saab0.fit(sample_patches)

            if i==0:
                transformed=saab0.transform(sample_patches,addBias=False)
            else:
                pca_params['Layer_%d/bias'%i]=saab0.Bias
                transformed=saab0.transform(sample_patches)
            
            # Reshape: place back as a 4-D feature map
            sample_images=transformed.reshape(num_samples, h, w,-1)

            # Maxpooling
            sample_images=block_reduce(sample_images, (1,2,2,1), np.max)

            print('Sample patches shape after flatten:', sample_patches.shape)
            print('Kernel shape:', saab0.Kernels.shape)
            print('Transformed shape:', transformed.shape)
            print('Sample images shape:', sample_images.shape)
            
            pca_params['Layer_%d/feature_expectation'%i]=saab0.Mean0
            pca_params['Layer_%d/kernel'%i]=saab0.Kernels
            pca_params['Layer_%d/pca_mean'%i]=saab0.pca.mean_

        return pca_params

    # Initialize
    def feature(self, sample_images, pca_params):

        num_layers=pca_params['num_layers']
        self.kernel_sizes=pca_params['kernel_size']

        for i in range(num_layers):
            print('--------stage %d --------'%i)
            # Extract parameters
            feature_expectation=pca_params['Layer_%d/feature_expectation'%i]
            kernels=pca_params['Layer_%d/kernel'%i]
            mean=pca_params['Layer_%d/pca_mean'%i]

            # Create patches
            sample_patches=self.window_process(sample_images,self.kernel_sizes[i],1) # overlapping
            h=sample_patches.shape[1]
            w=sample_patches.shape[2]
            # Flatten
            sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

            #saab0=saab.Saab(Kernels=kernels,trained=True,Mean0=feature_expectation,Bias=pca_params['Layer_%d/bias'%i])

            if i==0:
                saab0=saab.Saab(Kernels=kernels,trained=True,Mean0=feature_expectation)
                transformed=saab0.transform(sample_patches,addBias=False)
            else:
                saab0=saab.Saab(Kernels=kernels,trained=True,Mean0=feature_expectation,Bias=pca_params['Layer_%d/bias'%i])
                transformed=saab0.transform(sample_patches)
            
            # Reshape: place back as a 4-D feature map
            num_samples=sample_images.shape[0]
            sample_images=transformed.reshape(num_samples, h, w,-1)

            # Maxpooling
            sample_images=block_reduce(sample_images, (1,2,2,1), np.max)

            print('Sample patches shape after flatten:', sample_patches.shape)
            print('Kernel shape:', kernels.shape)
            print('Transformed shape:', transformed.shape)
            print('Sample images shape:', sample_images.shape)
        return sample_images