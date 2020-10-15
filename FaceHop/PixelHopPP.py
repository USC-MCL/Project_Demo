import numpy as np
import cv2
import os
from skimage.util import view_as_windows

from saab import Saab

class PixelHop_Unit():
    def __init__(self, X, num_kernels, window=5, stride=1):
        self.X = self.Shrink(X, window, stride)#N*28*28*(5*5*1)
        self.S = list(self.X.shape)
        self.X = self.X.reshape(-1, self.S[-1])#(N*28*28)*(5*5*1)
        self.num_kernels = num_kernels
        self.saab = None
        self.window = window
        self.stride = stride
    
    def train(self):
        self.saab = Saab(num_kernels=self.num_kernels, useDC=True)
        self.saab.fit(self.X)
    
    def transform(self,X):
        X = self.Shrink(X, self.window, self.stride)#N*28*28*(5*5*1)
        S = list(X.shape)
        X = X.reshape(-1, S[-1])#(N*28*28)*(5*5*1)
        assert (self.saab != None), "the model hasn't been trained, must call train() first!"
        transformed = self.saab.transform(X).reshape(S[0],S[1],S[2],-1)#N*28*28*25
        return self.saab,transformed

    def Shrink(self, X, win, stride):
        X = view_as_windows(X, (1,win,win,1), (1,stride,stride,1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

class PixelHopPP_Unit():
    def __init__(self, X, num_kernels, window=5, stride=1, energy_th=0, ch_decoupling=True, ch_energies=None):
        self.X = X
        self.N, self.L, self.W, self.D = self.X.shape#N*32*32*3
        if ch_energies == None:
            self.ch_energies = np.ones((self.D)).tolist()#[1,1,1]
        else:
            self.ch_energies = ch_energies
        self.out_ch_energies = []#25+25+25
        self.output = None
        self.kernel_filter = []
        self.energy_th = energy_th
        self.window = window
        self.stride = stride
        self.ch_decoupling = ch_decoupling
        self.num_kernels = num_kernels
        self.pixelHopUnit_list = []

    def train(self):
        if self.ch_decoupling == True:
            for i in range(self.D):
                pixelHopUnit = PixelHop_Unit(self.X[:,:,:,i].reshape(self.N,self.L,self.W,1), num_kernels=self.num_kernels, window=self.window, stride=self.stride)
                pixelHopUnit.train()
                saab, transformed = pixelHopUnit.transform(self.X[:,:,:,i].reshape(self.N,self.L,self.W,1))
                #transformed#N*28*28*25
                self.out_ch_energies.append(self.ch_energies[i] * saab.Energy)
                self.kernel_filter.append(self.out_ch_energies[i] > self.energy_th)#(25-a)+(25-b)+(25-c)
                transformed = transformed[:, :, :, self.kernel_filter[i]]#N*28*28*(25-a)
                if i == 0:
                    self.output = transformed
                else:
                    self.output = np.concatenate((self.output,transformed),axis=3)#N*28*28*((25-a)+(25-b)+(25-c))
                self.pixelHopUnit_list.append(pixelHopUnit)
        else:
            pixelHopUnit = PixelHop_Unit(self.X, num_kernels=self.num_kernels, window=self.window, stride=self.stride)
            pixelHopUnit.train()
            saab, transformed = pixelHopUnit.transform(self.X)
            self.out_ch_energies.append(saab.Energy)
            self.kernel_filter.append(self.out_ch_energies[0] > self.energy_th)
            transformed = transformed[:, :, :, self.kernel_filter[0]]
            self.output = transformed
            self.pixelHopUnit_list.append(pixelHopUnit)
        return self.flatten(self.out_ch_energies, self.kernel_filter)

    def transform(self, X):
        N, L, W, D = X.shape#N*32*32*3
        if self.ch_decoupling == True:
            for i in range(self.D):
                pixelHopUnit = self.pixelHopUnit_list[i]
                saab, transformed = pixelHopUnit.transform(X[:,:,:,i].reshape(N,L,W,1))
                #transformed#N*28*28*25
                transformed = transformed[:, :, :, self.kernel_filter[i]]#N*28*28*(25-a)
                if i == 0:
                    self.output = transformed
                else:
                    self.output = np.concatenate((self.output,transformed),axis=3)#N*28*28*((25-a)+(25-b)+(25-c))
        else:
            pixelHopUnit = self.pixelHopUnit_list[0]
            saab, transformed = pixelHopUnit.transform(X)
            
            transformed = transformed[:, :, :, self.kernel_filter[0]]
            self.output = transformed
        return self.output

    def flatten(self, listoflists, kernel_filter):#25+25+25 (25-a)+(25-b)+(25-c)
        flattened = []
        for i in range(len(listoflists)):
            for j in range(len(listoflists[i][kernel_filter[i]])):
                flattened.append(listoflists[i][j])
        return flattened