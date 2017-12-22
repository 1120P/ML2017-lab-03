import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import math
import pandas as pd
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):

    
        self.n_weakers_limit=10
        self.depth=3
        #弱分类器暂设为10，可在后期调整。
        self.n= 80*2 #400个样本 即x.shape[0]=80
        self.sample_weight= [1./self.n]*self.n#400个样本最开始赋予相同的权重
        
        self.a=[]#初始化参数
        self.alpha=1#初始化参数
        self.e=[0]* self.n_weakers_limit#初始化误差率e[m]
       
        self.pre=[]
                self.wclassifier=weak_classifier
        

        
        
        '''Initialize AdaBoostClassifier
            

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        
        
        
        pass


    def is_good_enough(self):
        '''Optional'''
        
        
        pass





    def fit(self,X,y):
  
        for m in range(self.n_weakers_limit):
        #rf=DecisionTreeClassifier(max_depth=3,min_samples_leaf=1,class_weight=samble_weight[m])#第m个基分类器
            self.wclassifier.fit(X, y, sample_weight=self.sample_weight)#训练m个分类器
            pre_y = self.wclassifier.predict(X).reshape(-1,1)
            y=y.reshape(-1,1)
            print(y)
            print(pre_y)
                #如果不正确加入误差率计算
            for i in range(self.n):
                if y[self.n:]!=pre_y[self.n:]:
                    self.e[m] += samble_weight[self.n] #误差率为错误的权重之和
            if self.e[m]==0:
                self.a.append(self.alpha)
                self.pre.append(self.wclassifier)
            else:
                alpha=0.5*math.log((1-self.e[m])/self.e[m]) #这是更新参数的参数
                self.a.append(self.alpha)
                Z=self.sample_weight.sum(dtype=np.float64)*exp(-alpha*self.pre_y*y)


                self.sample_weight=self.samble_weight.reshape(-1,1)*exp(-alpha*self.pre_y*y)/Z
                self.pre.append(self.wclassifier)
        
        
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
            
        '''
 

        
        pass






    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.
            Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        #f = sum([pre[i]*a[i] for i in range(self.n_weakers_limit)])
        
  


    
        pass

    def predict(self, X, threshold=0.5):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
            
        '''
        scores=np.empty(X.shape[0],dtype=np.float64).reshape(-1,1)
        
        for i in range(self.n_weakers_limit):
            p=np.concatenate((self.a[i]*self.pre[i].predict_proba(X)[:,1].reshape(-1,1)/(self.pre[i].predict_proba(X)[:,1].reshape(-1,1)+self.pre[i].predict_proba(X)[:, 0].reshape(-1, 1)),scores),axis=1)
     
        score=np.sum(p[:,0:-1],axis=1).reshape(-1,1)

        df = pd.DataFrame(score)
        
        df[1] = df[0].apply(lambda x: 1 if x < threshold else -1)
        df[2] = df[1].apply(lambda x: 1 if x < threshold else -1)
        
        
        return np.array(df[2]).reshape(-1,1)
        
    
        # Regression
   
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)



