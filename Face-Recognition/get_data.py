# Load imports
import os 
import cv2
import numpy as np

def get_images(image_directory,tNumber,ftc):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    
    print("Task selected: %s" % str(tNumber))
    taskNumber = "Task" + str(tNumber)

    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if tNumber < 10:
                    space = 5
                else:
                    space = 6
                
                if (ftc == 1):
                    if (file.endswith("FTC.png") and (taskNumber in file) and (file[space] == "_")) : # grab images only
                        print("Loaded %s" % file)
                        # read the image using openCV
                        img = cv2.cv2.imread(
                                os.path.join(image_directory, subfolder, file), cv2.cv2.IMREAD_GRAYSCALE
                                )
                        # resize the image by half
                        scale_percent = 50 # percent of original size
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img = cv2.cv2.resize(img, dim)
                        # add the resized image to a list X
                        X.append(img)
                        # add the image's label to a list y
                        y.append(subfolder)
                else:
                    if (("FTC" not in file) and (taskNumber in file) and (file[space] == "_")) : # grab images only
                        print("Loaded %s" % file)
                        # read the image using openCV
                        img = cv2.cv2.imread(
                                os.path.join(image_directory, subfolder, file), cv2.cv2.IMREAD_GRAYSCALE
                                )
                        # resize the image by half
                        scale_percent = 50 # percent of original size
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        img = cv2.cv2.resize(img, dim)
                        # add the resized image to a list X
                        X.append(img)
                        # add the image's label to a list y
                        y.append(subfolder)

    
    print("All images are loaded")     
    # return the images and their labels      
    return np.array(X), np.array(y)

# def get_landmarks(landmark_directory):
#     X = []
#     y = []
    
#     subfolders = os.listdir(landmark_directory)
#     for subfolder in subfolders:
#         print("Loading landmarks in %s" % subfolder)
#         if os.path.isdir(os.path.join(landmark_directory, subfolder)): # only load directories
#             subfolder_files = os.listdir(
#                     os.path.join(landmark_directory, subfolder)
#                     )
#             for file in subfolder_files:
#                 if file.endswith('5.npy'): 
#                     landmarks = np.load(os.path.join(landmark_directory, subfolder, file))
#                     X.append(landmarks)
#                     y.append(subfolder)
    
#     print("All landmarks are loaded")     
#     # return the images and their labels      
#     return np.array(X), np.array(y)
                    
                
            