# Load imports
import os 
import pandas as pd
import numpy as np

def get_keystrokes(keystroke_directory, k, k2):
    X1 = []
    y1 = []
    
    X2 = []
    y2 = []
    
    retrieve_headers = False
    subfolders = os.listdir(keystroke_directory)
    for subfolder in subfolders:
        # print("Loading keystroke data in %s" % subfolder)
        if os.path.isdir(os.path.join(keystroke_directory, subfolder)): 
            subfolder_files = os.listdir(
                    os.path.join(keystroke_directory, subfolder)
                    )
            for file in subfolder_files:
                if file == 'Session%d.csv' % k:
                  
                    session = pd.read_csv(os.path.join(keystroke_directory, subfolder, file))
                                    
                    if retrieve_headers == False:
                        headers = session.columns.tolist()
                        retrieve_headers = True
                        
                    X1.append(session.to_numpy())
                    y1.extend([subfolder] * len(session.index))
                    
            # ----------------------------------------
                    
            for file in subfolder_files:
                if file == 'Session%d.csv' % k2:
                  
                    session = pd.read_csv(os.path.join(keystroke_directory, subfolder, file))
                                          
                    X2.append(session.to_numpy())
                    y2.extend([subfolder] * len(session.index))
     
    return np.vstack(X1), np.array(y1), np.vstack(X2), np.array(y2), headers


            