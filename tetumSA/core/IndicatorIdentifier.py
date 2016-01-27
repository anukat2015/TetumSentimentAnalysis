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
import pandas as pd
from pprint import pprint
from nltk import FreqDist

class IndicatorIdentifier(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.model = LabelPropagation() #(kernel='knn', alpha=1.0)
        #self.model = LabelSpreading()
    
    def readingDatabase(self):
        da = DocumentsAccess()
        
        labeledFile = "Database/Indicator/Indicators.xlsx"
        sheet = "Sheet1"
        df = da.readingDatabaseTetum(labeledFile, sheet, head= 0)


        cut = int(0.8*df.shape[0])    
        # re-duplicate the data => Result: one document has one label only
        columns = df.columns.tolist()
        columns.remove("Content")
        print columns
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for index, row in df.iterrows():
            labels = list(set([row[col] for col in columns if not pd.isnull(row[col])]))
            content = row["Content"]
            if index < cut: # training part
                for label in labels:
                    X_train.append(content)
                    Y_train.append(label)
            else:
                X_test.append(content)
                Y_test.append(labels)
                
           
       
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
        
        '''
        fileUnlabeled4 = "Database/Haksesuk.xlsx"
        unlabeledData4 = da.readingDatabaseTetum(fileUnlabeled4, sheet)
        unlabeledData4 = unlabeledData4[0].tolist()
        print len(unlabeledData4)
        ''' 

        unlabeledData = unlabeledData + unlabeledData2 + unlabeledData3
        
        print len(unlabeledData)
        #print unlabeledData[0]
        
        return (X_train, Y_train, X_test, Y_test, unlabeledData) 
    
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
        (X_train, Y_train, X_test, Y_test, unlabeledData)   = self.readingDatabase()
        print "Training size: " + str(len(X_train))
        print "Test size: " + str(len(X_test))
        '''
        X_train = X_train[:100]
        Y_train = Y_train[:100]
        X_test = X_test[:100]
        Y_test = Y_test[:100]
        '''
        print "Finish reading database."
        #print FreqDist(indicators).most_common()
        k = 0
        dictLabel = FreqDist(Y_train)
        for key in dictLabel:
            dictLabel[key] = k
            k+=1
        Y_train = [dictLabel[ind] for ind in Y_train]
        Y_test = [[dictLabel[ind] for ind in labels] for labels in Y_test]
        
        '''       
        random.seed(123456)
        # Training
        z = zip(labeledData, indicators)
        random.shuffle(z)
        labeledData, indicators = zip(*z)
        
        X_train = list(labeledData[:cut])
        Y_train = list(indicators[:cut])
        X_test = list(labeledData[cut:])
        Y_test = list(indicators[cut:])
        '''
        X_train += unlabeledData
        Y_train += (-1*np.ones((len(unlabeledData)), dtype = int)).tolist()

        #pprint(X_train)
        #print Y_train
        
        #print X_train[cut-2:cut+2]
        #print Y_train[cut-2:cut+2]
        print "Training..."
        self.train(X_train, Y_train)
        
        # Testing
        print "Testing..."
        Y_predicted = self.test(X_test)

        print Y_predicted

        # The Y_predicted only need to be one of the true labels in order to be calculated as correctness
        for i in range(len(Y_predicted)):
            lab = Y_predicted[i]
            if lab in Y_test[i]:
                Y_test[i] = lab
            else:
                Y_test[i] = -1
        (accuracy,_, _, _) = self.evaluation(Y_test,Y_predicted)
        print accuracy  

    

#################################################################
#################################################################
#################################################################
def main():
    from datetime import datetime
    startTime = datetime.now()
    
        
    ii = IndicatorIdentifier()
    ii.run()
 
    elapsedTime = datetime.now() - startTime 
    totSecs = elapsedTime.total_seconds()
    theHours, reminder = divmod(totSecs, 3600)
    theMins, reminder = divmod(reminder, 60)
    theSecs, theMicroSec = divmod(reminder, 1)
    print "Time taken:  %02d hours, %02d minutes, %02d seconds" %(theHours, theMins, theSecs)               
        
if __name__ == "__main__":
    main()
