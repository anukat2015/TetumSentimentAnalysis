'''
Created on Dec 25, 2015

@author: Sony
'''
from __future__ import division
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from VectorSpaceModel import VectorSpaceModel
from tetumSA.databaseAccess.DocumentsAccess import DocumentsAccess
class SentimentAnalysis(object):
    '''
    classdocs
    '''


    def __init__(self, idDatabase, typeModel):
        self.arrDocuments = []
        self.model = linear_model.SGDClassifier()
    
    def readingDatabase(self):
        X, Y =  DocumentsAccess.readingDatabase()
        return X, Y
    
    def train(self):
        '''
            Goal: Batch training (Use only one time at the beginning. After that, use updateNewInformation function to update new information from new data)
        '''
        X, Y = self.readingDatabase()
        X = self.featureExtraction(X)
        self.model.fit(X, Y)    
    def test(self):
        '''
            Goal: predict a new document
        '''
        X, Y = self.readingDatabase()
        X = self.featureExtraction(X)
        self.model.predict(X, Y)            
    def updateNewInformation(self, x1, y1):
        '''
            Goal: Update the information from the new data (Online Learning)
            Run re-train model at weekend
        '''
        self.model.partial_fit(x1,y1)

    def featureExtractionTrain(self, X):
        self.vsm = VectorSpaceModel.createInstance("TFIDF")
        trainTable = self.vsm.train(X)
        return trainTable
    def featureExtractionPredict(self, X):
        testTable = self.vsm.test(X)
        return testTable

    def evaluationMethod(self, trueLabels, predictedLabels):
        accuracy = metrics.accuracy_score(trueLabels,predictedLabels)
        precision = metrics.precision_score(trueLabels,predictedLabels,average = None)
        recall = metrics.recall_score(trueLabels,predictedLabels,average = None)
        f1 = metrics.f1_score(trueLabels,predictedLabels, average=None)
        '''
        accuracy = round(accuracy,4)
        precision = round(precision,4)
        recall = round(recall,4)
        f1 = round(f1,4)
        '''
        result = [("Accuracy",accuracy),("Precision",precision),("Recall",recall),("f1",f1)]
        return result
        
       
    def run(self):

        self.readingDatabase()
        print "Finish reading database."

        ## Training
        self.train(self.arrDocuments)

    

#################################################################
#################################################################
#################################################################
def main():
    from datetime import datetime
    startTime = datetime.now()
    
        
    sentimentAnalysis = SentimentAnalysis()
    sentimentAnalysis.run()
 
    elapsedTime = datetime.now() - startTime 
    totSecs = elapsedTime.total_seconds()
    theHours, reminder = divmod(totSecs, 3600)
    theMins, reminder = divmod(reminder, 60)
    theSecs, theMicroSec = divmod(reminder, 1)
    print "Time taken:  %02d hours, %02d minutes, %02d seconds" %(theHours, theMins, theSecs)               
            
if __name__ == "__main__":
    main()
