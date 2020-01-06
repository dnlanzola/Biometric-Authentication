from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def classify(X1, y1, X2, y2, classifier, QUERY_SIZE, TEMPLATE_SIZE):
    
    if classifier == 1:          
        clf = KNeighborsClassifier()
    elif classifier == 2:
        clf = RandomForestClassifier()
    elif classifier == 3:
        clf = GaussianNB()
    else:
        clf = DecisionTreeClassifier()
        
    gen_scores = []
    imp_scores = []
    
    query_indices = np.random.choice(np.arange(0, len(y2), 1), int(QUERY_SIZE*len(y2)), replace=False)
    template_indices = np.random.choice(np.arange(0, len(y1), 1), int(TEMPLATE_SIZE*len(y1)), replace=False)
        
    for i in query_indices:
        try:
            query = X2[i, :]
            query_label = y2[i]
            
            templates = X1[template_indices, :]
            template_labels = y1[template_indices]
                
            # Set the appropriate labels
            # 1 is genuine, 0 is impostor
            y_hat = np.zeros(len(template_labels))
            y_hat[template_labels == query_label] = 1 
            y_hat[template_labels != query_label] = 0
            
            clf.fit(templates, y_hat) # Train the classifier
            scores = clf.predict_proba(query.reshape(1,-1)).reshape(1,2) # Predict the label of the query
            classes = clf.classes_.reshape(1,2)
            
            gen_scores.extend(scores[classes==1])
            imp_scores.extend(scores[classes==0])
        except:
            continue
        
    return gen_scores, imp_scores
        
        