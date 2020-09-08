# v2020.03.26
# PixelHop

import numpy as np 

from saab import Saab

def PixelHop_Neighbour(feature, dilate, pad):
    dilate = np.array([dilate]).reshape(-1)
    idx = [1, 0, -1]
    H, W = feature.shape[1], feature.shape[2]
    res = feature.copy()
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'constant', constant_values=0)
    elif pad == 'none':
        H, W = H - 2*dilate[-1], W - 2*dilate[-1]
        res = feature[:, dilate[-1]:dilate[-1]+H, dilate[-1]:dilate[-1]+W].copy()
    else:
        assert (False), "Error padding method! support 'reflect', 'zeros', 'none'."
    for d in range(dilate.shape[0]):
        for i in idx:
            for j in idx:
                if i == 0 and j == 0:
                    continue
                else:
                    ii, jj = (i+1)*dilate[d], (j+1)*dilate[d]
                    res = np.concatenate((feature[:, ii:ii+H, jj:jj+W], res), axis=3)
    return res 

def Batch_PixelHop_Neighbour(feature, dilate, pad, batch):
    if batch <= feature.shape[0]:
        res = PixelHop_Neighbour(feature[0:batch], dilate, pad)
    else:
        res = PixelHop_Neighbour(feature, dilate, pad)
    for i in range(batch, feature.shape[0], batch):
        if i+batch <= feature.shape[0]:
            res = np.concatenate((res, PixelHop_Neighbour(feature[i:i+batch], dilate, pad)), axis=0)
        else:
            res = np.concatenate((res, PixelHop_Neighbour(feature[i:], dilate, pad)), axis=0)
    return res

class Pixelhop():
    def __init__(self, dilate, pad, SaabArg, batch=None):
        self.saab = Saab(num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], needBias=SaabArg['needBias'])
        self.dilate = np.array([dilate]).tolist()
        self.pad = pad
        self.batch = batch
        self.trained = False
    
    def fit(self, X):
        if self.batch == None:
            X = PixelHop_Neighbour(X, self.dilate, self.pad)
        else:
            X = Batch_PixelHop_Neighbour(X, self.dilate, self.pad, self.batch)
        X = X.reshape(-1, X.shape[-1])
        self.saab.fit(X)
        self.trained = True
        return self
    
    def transform(self, X):
        assert (self.trained == True), "Call fit first!"
        if self.batch == None:
            X = PixelHop_Neighbour(X, self.dilate, self.pad)
        else:
            X = Batch_PixelHop_Neighbour(X, self.dilate, self.pad, self.batch)
        S = X.shape
        X = X.reshape(-1, X.shape[-1])
        X, DC = self.saab.transform(X)
        X = X.reshape(S[0], S[1], S[2], -1)
        return X, DC

if __name__ == "__main__":
    from sklearn import datasets
    
    # read data
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))
    SaabArg = {'num_AC_kernels':-1, 'needBias':False, 'useDC':True}

    # run
    hop1 = Pixelhop(dilate=1, pad='reflect', SaabArg=SaabArg, batch=None)
    hop1.fit(X)
    X2, DC = hop1.transform(X)
    print(" --> test feature shape: ", X2.shape)
    print("------- DONE -------\n")

