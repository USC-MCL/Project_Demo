import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags
import pickle
import keras
import sklearn
import SaabTrans
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

flags.DEFINE_string("output_path", None, "The output dir to save params")
flags.DEFINE_string("use_classes", "0-9", "Supported format: 0,1,5-9")
flags.DEFINE_string("kernel_sizes", "5,5", "Kernels size for each stage. Format: '3,3'")
flags.DEFINE_string("num_kernels", "5,15", "Num of kernels for each stage. Format: '4,10'")
flags.DEFINE_float("energy_percent", None, "Energy to be preserved in each stage")
flags.DEFINE_integer("use_num_images", -1, "Num of images used for training")
FLAGS = flags.FLAGS

class InterpretableCNN:
    def __init__(self,train_images, train_labels, test_images, test_labels, class_list):
        self.train_images=train_images
        self.train_labels=train_labels
        self.test_images=test_images
        self.test_labels=test_labels
        self.class_list=class_list
        self.pca_params='pca_params.pkl'
        self.transform=SaabTrans.SaabTrans(self.train_images, self.train_labels,
                            kernel_sizes=FLAGS.kernel_sizes,
                            num_kernels=FLAGS.num_kernels,
                            energy_percent=FLAGS.energy_percent,
                            use_num_images=FLAGS.use_num_images,
                            use_classes=self.class_list)
        self.gotKernel=False

    def getKernel(self):
        # read data
        #train_images, train_labels, test_images, test_labels, class_list = data.import_data(FLAGS.use_classes)
        print('Training image size:', self.train_images.shape)
        print('Testing_image size:', self.test_images.shape)

        #kernel_sizes=self.transform.parse_list_string(FLAGS.kernel_sizes)
        #if FLAGS.num_kernels:
    	#    num_kernels=self.transform.parse_list_string(FLAGS.num_kernels)
        #else:
    	#    num_kernels=None
        #energy_percent=FLAGS.energy_percent
        #use_num_images=FLAGS.use_num_images
        #print('Parameters:')
        #print('use_classes:', self.class_list)
        #print('Kernel_sizes:', kernel_sizes)
        #print('Number_kernels:', num_kernels)
        #print('Energy_percent:', energy_percent)
        #print('Number_use_images:', use_num_images)

        pca_params=self.transform.multi_Saab_transform()
        # save data
        fw=open(self.pca_params,'wb')    
        pickle.dump(pca_params, fw)    
        fw.close()

        # load data
        fr=open(self.pca_params,'rb')  
        data1=pickle.load(fr)
        print(data1)
        fr.close()

        self.gotKernel=True

    def getFeature(self):
        assert (self.gotKernel == True), "Must call getKernel first!"

        # load local data
        fr=open(self.pca_params,'rb')  
        pca_params=pickle.load(fr, encoding='latin')
        fr.close()

        # read data
        #train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
        print('Training image size:', self.train_images.shape)
        print('Testing_image size:', self.test_images.shape)
        
        # Training
        print('--------Training--------')
        feature=self.transform.feature(self.train_images, pca_params) 
        feature=feature.reshape(feature.shape[0],-1)
        print("S4 shape:", feature.shape)
        print('--------Finish Feature Extraction subnet--------')
        feat={}
        feat['feature']=feature
        
        # save data
        fw=open('feat.pkl','wb')    
        pickle.dump(feat, fw)    
        fw.close()

    def getWeight(self):
        

        # load local data
        fr=open(self.pca_params,'rb')  
        pca_params=pickle.load(fr, encoding='latin')
        fr.close()

        # read data
        #train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
        print('Training image size:', self.train_images.shape)
        print('Testing_image size:', self.test_images.shape)

        # load feature
        fr=open('feat.pkl','rb')  
        feat=pickle.load(fr, encoding='latin')
        fr.close()
        feature=feat['feature']
        print("S4 shape:", feature.shape)
        print('--------Finish Feature Extraction subnet--------')

        # feature normalization
        std_var=(np.std(feature, axis=0)).reshape(1,-1)
        feature=feature/std_var

        num_clusters=[120, 84, 10]
        use_classes=10
        weights={}
        bias={}
        for k in range(len(num_clusters)):
            if k!=len(num_clusters)-1:
                # Kmeans_Mixed_Class (too slow for CIFAR, changed into Fixed Class)
                kmeans=KMeans(n_clusters=num_clusters[k]).fit(feature)
                pred_labels=kmeans.labels_
                num_clas=np.zeros((num_clusters[k],use_classes))
                for i in range(num_clusters[k]):
                    for t in range(use_classes):
                        for j in range(feature.shape[0]):
                            if pred_labels[j]==i and self.train_labels[j]==t:
                                num_clas[i,t]+=1
                acc_train=np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
                print(k,' layer Kmean (just ref) training acc is {}'.format(acc_train))

                # Compute centroids
                clus_labels=np.argmax(num_clas, axis=1)
                centroid=np.zeros((num_clusters[k], feature.shape[1]))
                for i in range(num_clusters[k]):
                    t=0
                    for j in range(feature.shape[0]):
                        if pred_labels[j]==i and clus_labels[i]==self.train_labels[j]:
                            if t==0:
                                feature_test=feature[j].reshape(1,-1)
                            else:
                                feature_test=np.concatenate((feature_test, feature[j].reshape(1,-1)), axis=0)
                            t+=1
                    centroid[i]=np.mean(feature_test, axis=0, keepdims=True)

                # Compute one hot vector
                t=0
                labels=np.zeros((feature.shape[0], num_clusters[k]))
                for i in range(feature.shape[0]):
                    if clus_labels[pred_labels[i]]==self.train_labels[i]:
                        labels[i,pred_labels[i]]=1
                    else:
                        distance_assigned=euclidean_distances(feature[i].reshape(1,-1), centroid[pred_labels[i]].reshape(1,-1))
                        cluster_special=[j for j in range(num_clusters[k]) if clus_labels[j]==self.train_labels[i]]
                        distance=np.zeros(len(cluster_special))
                        for j in range(len(cluster_special)):
                            distance[j]=euclidean_distances(feature[i].reshape(1,-1), centroid[cluster_special[j]].reshape(1,-1))
                        labels[i, cluster_special[np.argmin(distance)]]=1

                # least square regression
                A=np.ones((feature.shape[0],1))
                feature=np.concatenate((A,feature),axis=1)
                weight=np.matmul(LA.pinv(feature),labels)
                feature=np.matmul(feature,weight)
                weights['%d LLSR weight'%k]=weight[1:weight.shape[0]]
                bias['%d LLSR bias'%k]=weight[0].reshape(1,-1)
                print(k,' layer LSR weight shape:', weight.shape)
                print(k,' layer LSR output shape:', feature.shape)

                pred_labels=np.argmax(feature, axis=1)
                num_clas=np.zeros((num_clusters[k],use_classes))
                for i in range(num_clusters[k]):
                    for t in range(use_classes):
                        for j in range(feature.shape[0]):
                            if pred_labels[j]==i and self.train_labels[j]==t:
                                num_clas[i,t]+=1
                acc_train=np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
                print(k,' layer LSR training acc is {}'.format(acc_train))

                # Relu
                for i in range(feature.shape[0]):
                    for j in range(feature.shape[1]):
                        if feature[i,j]<0:
                            feature[i,j]=0

                # # Double relu
                # for i in range(feature.shape[0]):
                # 	for j in range(feature.shape[1]):
                # 		if feature[i,j]<0:
                # 			feature[i,j]=0
                # 		elif feature[i,j]>1:
                # 			feature[i,j]=1
            else:
                # least square regression
                labels=keras.utils.to_categorical(self.train_labels,10)
                A=np.ones((feature.shape[0],1))
                feature=np.concatenate((A,feature),axis=1)
                weight=np.matmul(LA.pinv(feature),labels)
                feature=np.matmul(feature,weight)
                weights['%d LLSR weight'%k]=weight[1:weight.shape[0]]
                bias['%d LLSR bias'%k]=weight[0].reshape(1,-1)
                print(k,' layer LSR weight shape:', weight.shape)
                print(k,' layer LSR output shape:', feature.shape)
                
                pred_labels=np.argmax(feature, axis=1)
                acc_train=sklearn.metrics.accuracy_score(self.train_labels,pred_labels)
                print('training acc is {}'.format(acc_train))
        # save data
        fw=open('llsr_weights.pkl','wb')    
        pickle.dump(weights, fw)    
        fw.close()
        fw=open('llsr_bias.pkl','wb')    
        pickle.dump(bias, fw)    
        fw.close()