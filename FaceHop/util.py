# 2020.06.19
import numpy as np
from glob import glob
import cv2
import os
import sklearn
import sys
import csv
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from skimage.measure import block_reduce


def lfw_train_test(root, pair_txt_path, raw_images, flipped_raw_images, raw_labels, foldnum, includeflipped=False):
    with open(root+pair_txt_path, 'r') as csvfile:
        rows = list(csv.reader(csvfile, delimiter='\t'))[1:]
    kf = KFold(n_splits=10)
    fold = 0
    for train_index,test_index in kf.split(rows):
        fold += 1
        trainrows, testrows = [], []
        for i in train_index:
            trainrows.append(rows[i])
        for i in test_index:
            testrows.append(rows[i])
        if(fold==foldnum):
            break
    raw_labels_dic = {}
    for i in range(len(raw_labels)):
        if(includeflipped):
            raw_labels_dic[raw_labels[i]] = [raw_images[i], flipped_raw_images[i]]
        else:
            raw_labels_dic[raw_labels[i]] = [raw_images[i]]

    trainData1 = []#first image in a pair, even ones are origianl and odd ones are flipped versions
    trainData1flipped = []
    trainData2 = []#second image in a pair
    trainData2flipped = []
    trainLabel = []#match/mismatch label
    for row in trainrows:#testrows
        if len(row) == 3 :
            name1 = row[0] + '_' + format(int(row[1]), '04d')
            name2 = row[0] + '_' + format(int(row[2]), '04d')
            label = 1
        elif(len(row) == 4):
            name1 = row[0] + '_' + format(int(row[1]), '04d')
            name2 = row[2] + '_' + format(int(row[3]), '04d')
            label = 0#not the same
        flag = 0
        if includeflipped:
            if name1 in raw_labels_dic.keys():
                vect1 = raw_labels_dic[name1][0]
                vect2 = raw_labels_dic[name1][1]
                flag += 1
            if name2 in raw_labels_dic.keys():
                vect3 = raw_labels_dic[name2][0]
                vect4 = raw_labels_dic[name2][1]
                flag += 1
            if flag != 2:
                print("         <Warning> Train row: ", row," is not found!")
                continue
            trainData1.append(vect1)
            trainData1flipped.append(vect2)
            trainData2.append(vect3)
            trainData2flipped.append(vect4)
            trainLabel.append(label)
        else:
            if name1 in raw_labels_dic.keys():
                vect1 = raw_labels_dic[name1][0]
                flag += 1
            if name2 in raw_labels_dic.keys():
                vect3 = raw_labels_dic[name2][0]
                flag += 1
            if flag != 2:
                print("         <Warning> Train row: ", row," is not found!")
                continue
            trainData1.append(vect1)
            trainData2.append(vect3)
            trainLabel.append(label)

    testData1=[]#first image in a pair, even ones are origianl and odd ones are flipped versions
    testData1flipped=[]
    testData2=[]#second image in a pair
    testData2flipped=[]
    testLabel=[]#match/mismatch label
    for row in testrows:#testrows
        if len(row)==3:
            name1 = row[0] + '_' + format(int(row[1]), '04d')
            name2 = row[0] + '_' + format(int(row[2]), '04d')
            label = 1
        elif len(row)==4:
            name1 = row[0] + '_' + format(int(row[1]), '04d')
            name2 = row[2] + '_' + format(int(row[3]), '04d')
            label = 0#not the same
        flag = 0
        if name1 in raw_labels_dic.keys():
            vect1 = raw_labels_dic[name1][0]
            flag += 1
        if name2 in raw_labels_dic.keys():
            vect3 = raw_labels_dic[name2][0]
            flag += 1
        if flag != 2:
            print("         <Warning> Test row: ", row, " is not found!")
            continue
        testData1.append(vect1)
        testData2.append(vect3)
        testLabel.append(label)
    if(includeflipped):
        trainData1 = np.concatenate((trainData1,trainData1flipped), axis=0)
        trainData2 = np.concatenate((trainData2,trainData2flipped), axis=0)
        trainLabel = np.concatenate((trainLabel,trainLabel), axis=0)
    print("       <INFO> Get %s training pairs, %s testing pairs!"%(str(len(trainData1)), str(len(testData1))))
    return np.asarray(trainData1), np.asarray(trainData2), np.asarray(trainLabel), np.asarray(testData1), np.asarray(testData2), np.asarray(testLabel)

def myStandardScaler(X, S, train=True):
    shape = (X.shape[0], X.shape[1], X.shape[2])
    if train == True:
        S = []
        for i in range(X.shape[-1]):
            ss = StandardScaler()
            ss.fit(X[:,:,:,i].reshape(-1, 1))
            S.append(ss)
    for i in range(X.shape[-1]):
        tmp = S[i].transform(X[:,:,:,i].reshape(-1, 1))
        X[:,:,:,i] = tmp.reshape(shape)
    return X, S

def Generate_feature(x1, x2, hop=1, loc={'1': [[0,0, 10 ,12],[0, 16, 10, 28], [7, 9, 18, 19],[17,5, 25, 23]],
                                         '2':[[0,0, 4, 10], [4,1,10,9]],
                                         '3':10}):
    print("change")
    cos, ra = [], []
    if hop == 3:
        n = loc['3']
        for l in range(1, x1.shape[-1]//n+1):
            tmp = []
            rra = []
            for i in range(x1.shape[0]):
                vect1 = x1[i, :, :, n*(l-1):n*(l)].reshape(1, -1)
                vect2 = x2[i, :, :, n*(l-1):n*(l)].reshape(1, -1)                
                a = np.sqrt(np.sum(np.square(vect1)))
                b = np.sqrt(np.sum(np.square(vect2)))
                if a > b:
                    c = a/b
                else:
                    c = b/a
                rra.append(c)
                tmp.append([sklearn.metrics.pairwise.cosine_similarity(vect1,vect2)[0,0]])
            rra = np.array(rra).reshape(-1,1)
            ra.append(rra)
            tmp = np.array(tmp).reshape(-1, 1)
            cos.append(tmp)
        ra = np.concatenate(ra, axis=1)
        ra = np.mean(ra, axis=1, keepdims=True)
        cos = np.concatenate(cos, axis=1)
        print("         <INFO> Hop3 contributes %s cosine simality, %s length ratio!"%(str(cos.shape[-1]), str(ra.shape[-1])))
        return np.concatenate((ra, cos), axis=1)
    for l in loc[str(hop)]:
        tmp = []
        feature_by_space = []
        # -1 denotes principal components, 0 denotes training pairs
        for i in range(x1.shape[-1]):
            tmpp = []
            ra = []
            features = []
            for j in range(x1.shape[0]):
                vect1 = x1[i, l[0]:l[2], l[1]:l[3], j].reshape(1, -1)
                vect2 = x2[i, l[0]:l[2], l[1]:l[3], j].reshape(1, -1)
                features.append(vect1)
                features.append(vect2)
                a = np.sqrt(np.sum(np.square(vect1)))
                b = np.sqrt(np.sum(np.square(vect2)))
                if a > b:
                    c = a/b
                else:
                    c = b/a
                ra.append(c)
                tmpp.append([sklearn.metrics.pairwise.cosine_similarity(vect1,vect2)[0,0]])
            features = np.concatenate(features)
            print(features.shape)
            feature_by_space.append(features)
            print(feature_by_space)
            tmpp.append([np.mean(ra)])
            tmp.append(np.array(tmpp).reshape(-1))
        tmp = np.array(tmp)
        cos.append(tmp)
    print("         <INFO> Hop%s contributes %s cosine simality, %s length ratio!"%(str(hop), str(tmp.shape[-1]-len(loc[str(hop)])), str(len(loc[str(hop)]))))
    return np.concatenate(cos, axis=1)

def MaxPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.max)

def get_gender_label(folder_path):
    image_file_name = []
    for fullfile in glob(folder_path+'/data/HEFrontalizedLfw2/*'):
        image_file_name.append(fullfile.split('/')[-1].strip('.jpg\n'))
    female_names = []
    with open(folder_path+'/data/female_names.txt') as f:
        content = f.readlines()
        for line in content:
            female_names.append(line.strip('.jpg\n'))
    male_names = []
    with open(folder_path+'/data/male_names.txt') as f:
        content = f.readlines()
        for line in content:
            male_names.append(line.strip('.jpg\n'))
    gender_label = []
    for i, name in enumerate(image_file_name):
        if name in female_names:
            gender_label.append(0)
        elif name in male_names:
            gender_label.append(1)
        else:
            print(name, "label not found!")
    return np.array(gender_label)   

def get_image_array(folder_path):
    image_list = []
    for filename in glob(folder_path+'/data/HEFrontalizedLfw2/*'):
        image = cv2.imread(filename)
        image = cv2.resize(image, (32,32))
        image_list.append(image/255)
    image_array = np.array(image_list)
    return image_array