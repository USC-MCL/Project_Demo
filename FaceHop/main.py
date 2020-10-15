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
#from PixelHopPP import PixelHopPP_Unit
from util import get_gender_label, get_image_array, myStandardScaler, Generate_feature, MaxPooling
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
import FaceHop
warnings.filterwarnings("ignore")

n1 =18
n2 = 13
n3 = 11
numComponents = 20
standardize = False
energy_th = 0.0005
num_of_train_pixelhop = 4000
foldnum = 1


def LR(x_train, y_train, x_test, y_test):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print("     <INFO> train acc: %s"%(clf.score(x_train, y_train)))
    print("     <INFO> test acc: %s"%(clf.score(x_test, y_test)))
    return clf.predict_proba(x_train), clf.predict_proba(x_test)

def SVM(x_train, y_train, x_test, y_test):
    clf = SVC(gamma='auto', probability=True)
    clf.fit(x_train, y_train)
    print("     <INFO> train acc: %s"%(clf.score(x_train, y_train)))
    print("     <INFO> test acc: %s"%(clf.score(x_test, y_test)))
    return clf.predict_proba(x_train), clf.predict_proba(x_test)


def data_aug(xf):
    print('before aug', xf.shape)
    x1 = flip_aug(xf)
    x2 = pca_aug(xf)
    xf = np.concatenate((xf, x1, x2), axis=0)
    print('after aug', xf.shape)
    return xf

def cwPCA(x, eng_percent):
    pca = PCA(n_components=500)
    x = pca.fit_transform(x)
    ratio = np.cumsum(pca.explained_variance_ratio_) >= eng_percent
    n_comp = np.argmax(ratio)
    #print(n_comp, " compontents retained!")
    x = x[:, :n_comp]
    dis = euclidean_distances(x, x)+1000*np.eye(len(x))
    return dis

def pca_aug(x, eng_percent=0.9):
    xx = x.copy()
    x = x.reshape(x.shape[0], -1, 3)/255
    dis0 = cwPCA(x[:,:,0], eng_percent)
    dis1 = cwPCA(x[:,:,1], eng_percent)
    dis2 = cwPCA(x[:,:,2], eng_percent)
    idx = np.argmin(dis0+dis1+dis2, axis=1)
    new_x = []
    ct = 1
    for i in range(len(xx)):
        tmp = xx[i]/2 + xx[idx[i]] / 2
        #tmp = cv2.equalizeHist(tmp.astype(np.uint8))
        new_x.append(tmp)
        if ct > 0:
            plt.imshow(tmp[:,:])
            plt.title('mean')
            plt.show()
            plt.imshow(xx[i,:,:])
            plt.title('raw')
            plt.show()
            ct -= 1
    return np.array(new_x).reshape(xx.shape[0], xx.shape[1], xx.shape[2], -1)

def flip_aug(x):
    new_x = []
    for i in range(len(x)):
        new_x.append(cv2.flip(x[i], 1).reshape(x.shape[1], x.shape[2], -1))
    return np.array(new_x)

def main():
    #folder_path = dirname(getcwd())
    folder_path = getcwd()
    lfwlabels = get_gender_label(folder_path)
    lfw_raw = get_image_array(folder_path)
    train_images, test_images, y, yt = train_test_split(lfw_raw, lfwlabels, test_size=0.2, stratify=lfwlabels)
    print(train_images.shape, y.shape)

    x_aug = data_aug(train_images[y==0])
    train_images = np.concatenate((train_images, x_aug), axis=0)
    y = np.concatenate((y, np.zeros(len(x_aug))), axis=0)
    print(train_images.shape, y.shape)

    faceHop = FaceHop.FaceHop(train_images, test_images, n1, n2, n3, numComponents, energy_th, num_of_train_pixelhop)
    x, xt = faceHop.feature_extraction()

    px, pxt = [], []
    for i in range(len(x)):
        for j in range(len(x[i])):
            a, b = LR(x[i][j], y, xt[i][j], yt)
            px.append(a)
            pxt.append(b)
        print()
    print('\n ensemble')
    px = np.concatenate(px, axis=1)
    pxt = np.concatenate(pxt, axis=1)
    print('')
    a, b = LR(px, y, pxt, yt)

    print('LR all')
    a, b = LR(px[:,:], y, pxt[:,:], yt)
    print('LR hop2+hop3')
    a, b = LR(px[:,8:], y, pxt[:,8:], yt)
    print('SVM all')
    a, b = SVM(px[:,:], y, pxt[:,:], yt)
    print('SVM hop2+hop3')
    a, b = SVM(px[:,8:], y, pxt[:,8:], yt)

    #with open(getcwd()+'/face_gender_fea_'+str(time.time())+'.pkl', 'wb') as f:
        #pickle.dump({'x':x, 'xt':xt, 'y':y, 'yt':y, 'px':px, 'pxt':pxt},f)

if __name__ == '__main__':
	main()
