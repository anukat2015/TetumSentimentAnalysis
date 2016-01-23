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
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
import random
import TetumPreprocessing as tp
class SentimentAnalysis(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.arrDocuments = []
        self.model = LabelPropagation()
        #self.model = LabelSpreading()
    
    def readingDatabase(self):
        da = DocumentsAccess()
        
        filePos = "Database/Sentiment/Sentiment/PoliceRelations/positive.xlsx"
        sheet = "Sheet1"
        posData = da.readingDatabaseTetum(filePos, sheet)
        posData = posData[0].tolist()
        print len(posData)
        print posData[0]

        
 
        fileNeg = "Database/Sentiment/Sentiment/PoliceRelations/negative.xlsx"
        sheet = "Sheet1"
        negData = da.readingDatabaseTetum(fileNeg, sheet)  
        negData = negData[0].tolist() 
        print len(negData)
        print negData[0]
       
        fileUnlabeled = "Database/Clean Master Cleaner 2222.xlsx"
        sheet = "Sheet1"
        unlabeledData = da.readingDatabaseTetum(fileUnlabeled, sheet)
        unlabeledData = unlabeledData[0].tolist() 
        print len(unlabeledData)
        
        fileUnlabeled2 = "Database/SAPO.xlsx"
        unlabeledData2 = da.readingDatabaseTetum(fileUnlabeled2, sheet)
        unlabeledData2 = unlabeledData2[0].tolist() 
        print len(unlabeledData2)
        
        fileUnlabeled3 = "Database/Suara News.xlsx"
        unlabeledData3 = da.readingDatabaseTetum(fileUnlabeled3, sheet)
        unlabeledData3 = unlabeledData3[0].tolist() 
        print len(unlabeledData3)
        unlabeledData = unlabeledData + unlabeledData2 + unlabeledData3
        
        print len(unlabeledData)
        print unlabeledData[0]

       
        return (posData, negData, unlabeledData) 
    
    def preprocessData(self, X):   
        return tp.preprocess_dataset(X, fold=True, specials=False, min_size=2)       
    def train(self, X, Y):
        '''
            Goal: Batch training (Use only one time at the beginning. After that, use updateNewInformation function to update new information from new data)
        '''
        X = self.preprocessData(X)
        X = self.featureExtractionTrain(X)
        X = X.toarray()
        self.model.fit(X, Y)    
    def test(self, X):
        '''
            Goal: predict a new document
        '''
        X = self.preprocessData(X)
        X = self.featureExtractionPredict(X)
        X = X.toarray()
        predictedY = self.model.predict(X)   
        return predictedY      
       
    def updateNewInformation(self, x1, y1):
        '''
            Goal: Update the information from the new data (Online Learning)
            Run re-train model at weekend
        '''
        #self.model.partial_fit(x1,y1)
        pass

    def featureExtractionTrain(self, X):
        self.vsm = VectorSpaceModel.createInstance("TFIDF")#("BooleanWeighting") #("TFIDF")
        trainTable = self.vsm.train(X)
        return trainTable
    
    def featureExtractionPredict(self, X):
        testTable = self.vsm.test(X)
        return testTable

    def evaluation(self, trueLabels, predictedLabels):
        accuracy = metrics.accuracy_score(trueLabels,predictedLabels)
        precision = metrics.precision_score(trueLabels,predictedLabels,pos_label=None, average='weighted')
        recall = metrics.recall_score(trueLabels,predictedLabels,pos_label=None, average='weighted')
        f1 = metrics.f1_score(trueLabels,predictedLabels,pos_label=None, average='weighted')
        accuracy = round(accuracy,4)
        precision = round(precision,4)
        recall = round(recall,4)
        f1 = round(f1,4)
        result = [("Accuracy",accuracy),("Precision",precision),("Recall",recall),("f1",f1)]
        return result    
       
    def run(self):
        # Reading data
        (posData, negData, unlabeledData)  = self.readingDatabase()
        print "Finish reading database."
           
        # Divide training and test data
        cut = 10
        posDataTrain = posData[:cut]
        negDataTrain = negData[:cut*4]
        posDataTest = posData[cut:]
        negDataTest = negData[cut*4:]        
        
        random.seed(123456)
        # Training
        X_train = posDataTrain + negDataTrain + unlabeledData
        Y_train = np.ones((len(posDataTrain)), dtype = int).tolist() + np.zeros((len(negDataTrain)), dtype = int).tolist() + (-1*np.ones((len(unlabeledData)), dtype = int)).tolist()
        z = zip(X_train, Y_train)
        random.shuffle(z)
        X_train, Y_train = zip(*z)
        self.train(X_train, Y_train)
        
        # Testing
        X_test = posDataTest + negDataTest
        Y_test = np.ones((len(posDataTest)), dtype = int).tolist() + np.zeros((len(negDataTest)), dtype = int).tolist()
        z = zip(X_test, Y_test)
        random.shuffle(z)
        X_test, Y_test = zip(*z)
        Y_predicted = self.test(X_test)
        print Y_predicted

        (accuracy,precision,recall,f1) = self.evaluation(Y_test,Y_predicted)
        print (accuracy,precision,recall,f1)  

    

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
