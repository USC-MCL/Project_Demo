# 2020.06.05
# activate learning methods
# @yifan
import sys
import numpy as np 
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import time

def entropy_query(prob, num_feature):
    #prob is the output of the clf.predict_proba on the pool of unlabeled data
    #num_feature is the number of images you want to select using the entropy method
    entropies = np.zeros(prob.shape[0])
    for i in range(prob.shape[0]):
        entropies[i] = np.sum(-prob[i]*np.log(prob[i]+.0000001))
    th = np.sort(entropies)[prob.shape[0]-num_feature]
    num_feature_idx = entropies <= th
    num_feature_idx = entropies >= th
    return num_feature_idx

def coreset(x_pool, x_train,k):
    #x_pool is the pool of unlabeled data
    #x_train is the current set of labeled data
    #k is the number of images to select from x_pool
    dists = distance.cdist(x_pool, x_train, 'euclidean')
    nearesttolabeleds = np.min(dists,axis=1)
    th = np.sort(nearesttolabeleds)[nearesttolabeleds.shape[0]-k]
    num_feature_idx = nearesttolabeleds >= th
    return num_feature_idx

class QBC():
    def __init__(self, learners, init=0.01, n_increment=200, n_iter=40, percent=0.05):
        self.init = init
        self.n_increment = n_increment
        self.n_learner = len(learners)
        self.n_iter = n_iter
        self.num_class = 3
        self.learners = learners
        self.percent = percent
        self.trained = False
        self.acc_t = []
        self.acc_v = []
    
    def metric(self, prob):
        return entropy(prob, base=self.num_class, axis=1)

    def fit(self, x, y, xv=None, yv=None):
        self.trained = True
        self.num_class = np.unique(y).shape[0]
        #x, xt, y, yt = train_test_split(x, y, train_size=self.init, random_state=42, stratify=y)
        idx = np.random.choice(x.shape[0], (int)(x.shape[0]*self.percent))
        x_train, y_train = x[idx], y[idx]
        x_pool = np.delete(x, idx, axis=0)
        y_pool = np.delete(y, idx, axis=0)
        acc_t, acc_v, s = [], [], []
        for k in range(self.n_iter):
            print('       start iter -> %3s'%str(k))
            t0 = time.time()
            for i in range(self.n_learner):
                self.learners[i].fit(x_train, y_train)
            pt = self.predict_proba(x_pool)
            at = accuracy_score(y_pool, np.argmax(pt, axis=1))
            acc_t.append(at)
            s.append(y_pool.shape[0])
            ht = self.metric(pt)
            try:
                xv.shape
                print('           test shape: %s, val shape: %s'%(str(x_pool.shape), str(xv.shape)))
                pv = self.predict_proba(xv)
                av = accuracy_score(yv, np.argmax(pv, axis=1))
                print('           <Acc> test: %s, val: %s'%(at, av))
                acc_v.append(av)
                hv = self.metric(pv)
                print('           <Entropy> test: %s, val: %s'%(np.mean(ht), np.mean(hv)))
            except:
                pass
            idx = np.argsort(ht)[-self.n_increment:]
            x_train = np.concatenate((x_train, x_pool[idx]), axis=0)
            y_train = np.concatenate((y_train, y_pool[idx]), axis=0)
            x_pool = np.delete(x_pool, idx, axis=0)
            y_pool = np.delete(y_pool, idx, axis=0)
            print('       end iter -> %3s using %10s seconds\n'%(str(k),str(time.time()-t0)))
        self.acc_t = acc_t
        self.acc_v = acc_v
        return s, acc_t, acc_v

    def predict_proba(self, x):
        assert (self.trained == True), "Must call fit first!"
        pred = np.zeros((x.shape[0], self.num_class))
        for i in range(self.n_learner):
            pred += self.learners[i].predict_proba(x)
        return pred / np.sum(pred, axis=1, keepdims=True)
    
    def predict(self, x):
        assert (self.trained == True), "Must call fit first!"
        pred = self.predict_proba(x)
        return np.argmax(pred, axis=1)
    
    def score(self, x, y):
        assert (self.trained == True), "Must call fit first!"
        pred = self.predict(x)
        return accuracy_score(y, pred)