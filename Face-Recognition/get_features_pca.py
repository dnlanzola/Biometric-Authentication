import numpy as np

def pca(X):
    
    # Squash it
    faces = np.zeros(shape=(X.shape[1]*X.shape[2], X.shape[0]))
    for i in range(X.shape[0]):
        faces[:,i] = X[i].reshape(X.shape[1]*X.shape[2])
    
    # Get the mean face
    mean_face = faces.mean(axis=1)
       
    # Subtract the mean face - center everybody
    for col in range(faces.shape[1]):
        faces[:,col] = faces[:,col] - mean_face         
    
    # Compute the covariance matrix, C
    C = np.cov(faces.transpose())
        
    # Get the eigenfaces from C
    evals, evecs = np.linalg.eig(C)
    
    # Get eigenfaces
    eigenfaces = np.dot(faces, evecs)
              
    # get top k pca components
    k = 5
    face_features = np.zeros((faces.shape[1], k))
    for i in range(faces.shape[1]):
        face = faces[:,i]
        for j in range(k):
            face_features[i,j] = np.dot(eigenfaces[:,j].transpose(), face)
            
    return face_features



# def get_lbp(image, width):
#     image = image.reshape((width, width))
#     lbp_image = np.zeros(shape=(width, width))
#     num_neighbors = 1
    
#     for i in range(num_neighbors, image.shape[0] - num_neighbors):
#         for j in range(num_neighbors, image.shape[1] - num_neighbors):
#             center_pixel = image[i,j]
#             binary_string = ""
            
#             for m in range(i-num_neighbors, i+num_neighbors+1):
#                 for n in range(j-num_neighbors, j+num_neighbors+1):
                    
#                     if [i,j] == [m,n]: # same pixel
#                         pass
#                     else:
#                         neighbor_pixel = image[m, n]                                           
#                         if center_pixel >= neighbor_pixel:
#                             binary_string += '1'
#                         else:
#                             binary_string += '0'
            
#             lbp_image[i,j] = int(binary_string, 2)            
#     return lbp_image



