'''
Created on Oct 7, 2014

@author: thiennguyen
'''

import glob, os
import numpy as np
import random


#----------------------------------------------------------------------
class DocumentsAccess():   
    # Return array of sentences in the directory
    @staticmethod
    def parseFromDirectory(directoryPath):
        arrDocuments = []
        for name in os.listdir(directoryPath):
            if os.path.isfile(os.path.join(directoryPath, name)):
                with open(os.path.join(directoryPath, name)) as myfile:
                    document = myfile.read()
                    arrDocuments.append(document)
        return arrDocuments     
    
    @staticmethod
    def readingDatabase():
        posDocs = DocumentsAccess.parseFromDirectory('Database/txt_sentoken/pos/')
        negDocs = DocumentsAccess.parseFromDirectory('Database/txt_sentoken/neg/')
        posLabels = np.ones(len(posDocs)).tolist()
        negLabels = (-1 * np.ones(len(negDocs))).tolist()
        allDocs = posDocs + negDocs
        allLabels = posLabels + negLabels
        
        z = zip(allDocs, allLabels)
        random.seed(12345)
        random.shuffle(z)
        allDocs, allLabels = zip(*z)
        return allDocs, allLabels
            
            


#----------------------------------------------------------------------
if __name__ == "__main__":
    allDocs, allLabels = DocumentsAccess.readingDatabase()
    print allLabels[:10]
    
