import get_data
import matcher
import performance 
import enhance
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

''' Load the data and their labels '''
keystroke_directory = 'Project 2 Data'

eers = []
xlabels = []

for session1 in range(1, 9):
    for session2 in range(session1 + 1, 9):
        print("Session %d vs Session %d" % (session1, session2))
        xlabels.append("Session %d vs Session %d" % (session1, session2))
        template, templateLabels, query, queryLabels, events = get_data.get_keystrokes(keystroke_directory, session1, session2)
        
        # Could reduce X to only certain events (or columns) here
        
        ''' Feature enhancement/selection '''
        template, query = enhance.enhancement(template, query, 1)
              
        ''' Matching with chosen classifier'''
        qs = 0.10
        ts = 0.10
        gen_scores, imp_scores = matcher.classify(template, templateLabels,
                                                  query, queryLabels,
                                                  2, qs, ts)
        
        ''' Performance assessment '''
        eers.append(performance.perf(gen_scores, imp_scores, session1, session2))
        
plt.figure()
plt.plot(np.arange(0, len(eers), 1), eers, lw=2)        
plt.xlabel('Session Comparison')
plt.ylabel('EERs')
plt.xticks(np.arange(0, len(eers), 1), xlabels, rotation=90)
plt.title("Equal Error Rates")
plt.show()
plt.savefig('EERS.png', bbox_inches="tight")
        
    

