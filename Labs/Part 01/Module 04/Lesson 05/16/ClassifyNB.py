from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    cls = GaussianNB()
    return cls.fit(features_train, labels_train)
    
    
    ### your code goes here!
    
    