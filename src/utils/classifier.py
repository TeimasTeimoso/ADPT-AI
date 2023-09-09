from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class ClfSwitcher(BaseEstimator):
    
    def __init__(self, estimator=RandomForestClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        
        Parameters
        ----------
        estimator: sklearn object, the classifier
        """
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)