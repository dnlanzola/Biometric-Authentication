#######################################################
#
#   To run the code simply execute the main file.
#   The code will iterate through all the subjects.
#   
#   To run kNN set the 'match' variable to 1
#   To run Naïve Bayes set the 'match' variable to 2
#   To run PCA set the 'feature' variable to 1 
#   To run LBP set the 'feature' variable to 2
#
#   To choose whether you want FTC or NFTC set
#   the variable imgType to either 0 for NFTC 
#   and 1 for FTC.
#
#######################################################
import get_data
import get_features_pca
import get_features_lbp
import matcher
import performance 
import numpy as np

''' Load the data and their labels '''
image_directory = 'Project Data'

# This variable chooses the classifier: 1 = kNN / 2 = Naïve Bayes
match = 2

# This variable chooses either PCA or LBP: 1 = PCA / 2 = LBP
features = 1

# Type of images 0 = NFTC / 1 = FTC
imgType = 0

width = 50
blockSize = 16

for i in range(1,25):


    X, y = get_data.get_images(image_directory,i,imgType)

    y_list = y.tolist()
    if y_list.count('D_January') > 2:
        if y_list.count('R_Miller') > 2:
            if y_list.count('D_Anzola') > 2:
                if y_list.count('S_Ali') > 2:
                    if y_list.count('A_Sarker') > 2:

                        imgCount = y_list.count('D_January') + y_list.count('R_Miller') + y_list.count('D_Anzola') + y_list.count('S_Ali') + y_list.count('A_Sarker')



                        if X.shape[0] > 0:

                            if features == 1:
                                ''' Get PCA components '''
                                X = get_features_pca.pca(X)

                            if features == 2:
                                ''' Get LBP '''
                                featuresLbp = []
                                for k in range(imgCount):
                                    lbp_face = get_features_lbp.get_lbp(X[k], width)
                                    lbp_features = get_features_lbp.get_features_lbp(lbp_face, blockSize)
                                    featuresLbp.append(lbp_features)
                                X = np.array(featuresLbp)

                            
                            gen_scores, imp_scores = matcher.knn(X, y, match)




                            ''' Performance assessment '''
                            performance.perf(gen_scores, imp_scores, i)

    else:
        print("Not enough FTCs in Task %d" % i)

