from sklearn import preprocessing 
from sklearn.decomposition import PCA

def enhancement(template, query, k):
    
    if k == 1:
        ss = preprocessing.StandardScaler()
        ss.fit(template)
        template = ss.transform(template)
        query = ss.transform(query)
        
    elif k == 2:
        rs = preprocessing.RobustScaler()
        rs.fit(template)
        template = rs.transform(template)
        query = rs.transform(query) 
        
    elif k == 3:
        mm = preprocessing.MinMaxScaler()
        mm.fit(template)
        template = mm.transform(template)
        query = mm.transform(query)           
        
    elif k == 4:
        pca = PCA(n_components=5)
        pca.fit(template)
        template = pca.transform(template)
        query = pca.transform(query)
        
    else:
        print("No enhancement applied. Returning original data.")
        
    return template, query
