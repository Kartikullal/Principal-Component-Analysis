import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

class PCA_(TransformerMixin):
    
    def __init__(self, n_components = None):
        
        self.n_components = n_components
    
    def fit(self, X):
        self.X = X
        if(self.n_components == None):
            self.n_components = min(self.X.shape[0], self.X.shape[1])
        
        self.X = X - np.mean(X , axis = 0)
        
        self.cov_mat = np.cov(self.X , rowvar = False)
        
        self.eigen_values , self.eigen_vectors = np.linalg.eigh(self.cov_mat)
        
        sorted_index = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[sorted_index]
        self.eigen_vectors = self.eigen_vectors[:,sorted_index]
        
        self.eigenvector_subset = self.eigen_vectors[:,0:self.n_components]
    
    def transform(self, X):
        X_reduced = np.dot(self.eigenvector_subset.transpose() , self.X.transpose() ).transpose()
        
        return X_reduced