import numpy as np
import sklearn
import pickle
import sys
from os.path import dirname
from os import getcwd
from sklearn.svm import SVC
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import PixelHopPP
import cv2
from sklearn import preprocessing
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from numpy.random import RandomState
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import warnings
from glob import glob
from skimage.measure import block_reduce
warnings.filterwarnings("ignore")


class FaceHop():
    def __init__(self, X_train, X_test, n1 = 18, n2 = 13, n3 = 11, numComponents = 20, energy_th = 0.0005, num_of_train_pixelhop = 4000):
        self.X_train = X_train
        self.X_test = X_test
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n_components = numComponents
        self.energy_th = energy_th
        self.num_of_train_pixelhop = num_of_train_pixelhop

        
    def feature_extraction(self):
        allpatches = np.concatenate((self.X_train, self.X_test),axis=0)
        S = [len(self.X_train), len(self.X_test)]

        pixelHopPP_l1 = PixelHopPP.PixelHopPP_Unit(allpatches[0:self.num_of_train_pixelhop], num_kernels=self.n1, window=5, stride=1, energy_th=self.energy_th, ch_decoupling=False, ch_energies=None)
        flattened = pixelHopPP_l1.train()
        out1 = pixelHopPP_l1.transform(allpatches)
        out1ave = self.MaxPooling(out1)
        print("       <INFO> Hop1 #Nodes: %s"%(out1.shape[-1]))

        pixelHopPP_l2 = PixelHopPP.PixelHopPP_Unit(out1ave[0:self.num_of_train_pixelhop], num_kernels=self.n2, window=5, stride=1, energy_th=self.energy_th, ch_decoupling=True, ch_energies=flattened)
        flattened = pixelHopPP_l2.train()
        out2 = pixelHopPP_l2.transform(out1ave)
        out2ave = self.MaxPooling(out2)
        print("       <INFO> Hop2 #Nodes: %s"%(out2.shape[-1]))

        pixelHopPP_l3 = PixelHopPP.PixelHopPP_Unit(out2ave[0:self.num_of_train_pixelhop], num_kernels=self.n3, window=5, stride=1, energy_th=self.energy_th, ch_decoupling=True, ch_energies=flattened)
        flattened = pixelHopPP_l3.train()
        out3 = pixelHopPP_l3.transform(out2ave)
        out3ave = self.MaxPooling(out3)
        print("       <INFO> Hop3 #Nodes: %s"%(out3.shape[-1]))

        out1_train, out1_test= out1[0:S[0]], out1[S[0]:S[1]+S[0]]
        out2_train, out2_test= out2[0:S[0]], out2[S[0]:S[1]+S[0]]
        out3_train, out3_test= out3[0:S[0]], out3[S[0]:S[1]+S[0]]

        
        out1_train, p = self.Generate_feature_single_img(out1_train, self.n_components, pca_list=[], hop=1)
        out1_test, _ = self.Generate_feature_single_img(out1_test, self.n_components, pca_list=p, hop=1)

        out2_train, p = self.Generate_feature_single_img(out2_train, self.n_components, pca_list=[], hop=2)
        out2_test, _ = self.Generate_feature_single_img(out2_test, self.n_components, pca_list=p, hop=2)

        out3_train = out3_train.reshape(out3_train.shape[0], -1)
        out3_test = out3_test.reshape(out3_test.shape[0], -1)  
        
        return [out1_train, out2_train, [out3_train]], [out1_test, out2_test, [out3_test]]

    def Generate_feature_single_img(self, x, n_comp, pca_list=[], hop=1, loc={'1': [[0, 0, 10 ,12],[0, 16, 10, 28], [7, 9, 18, 19], [17,5, 25, 23]],
                                    '2':[[0, 0, 3, 10], [6, 0, 10, 10], [0, 3, 10, 7]]}):
        # old '2':[[0, 0, 4, 10], [4, 1, 10, 9]]
        fea_in_loc = []
        lenn = len(pca_list)
        for i in range(len(loc[str(hop)])):
            l = loc[str(hop)][i]
            tmp_fea = []
            tmp_pca = []
            for k in range(x.shape[-1]):
                tmp = x[:, l[0]:l[2], l[1]:l[3], k].reshape(x.shape[0], -1)
                if lenn == 0:
                    pca = PCA(n_components=n_comp)
                    pca.fit(tmp)
                    tmp_pca.append(pca)
                else:
                    pca = pca_list[i][k]
                tmp = pca.transform(tmp)
                tmp_fea.append(tmp)
            fea_in_loc.append(np.concatenate(tmp_fea, axis=1))
            if lenn == 0:
                pca_list.append(tmp_pca)
        return fea_in_loc, pca_list

    def MaxPooling(self, x):
        return block_reduce(x, (1, 2, 2, 1), np.max)