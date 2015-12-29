'''
Created on Nov 26, 2013

@author: thiennguyen
'''

from sklearn import svm
from VectorSpaceModel import VectorSpaceModel
from sklearn import metrics
class DocumentClassification:
    def run(self,iTrainListDocuments, iTrainLabels, iTestListDocuments, iTestLabels):
        self.train(iTrainListDocuments, iTrainLabels)
        predictedLabels = self.test(iTestListDocuments)    
        (accuracy,precision,recall,f1) = self.evaluation(iTestLabels,predictedLabels)
        return (accuracy,precision,recall,f1)        
    def train(self,iListDocuments,iListLabels):
        ''' Input:
                iListDocuments: list (list of documents, each document is a list of features: document1 = [ feature1, feature2, ... featureN])
                iListLabels: list (list of labels)
        '''
        self.vsm = VectorSpaceModel.createInstance("TFIDF")
        trainTable = self.vsm.train(iListDocuments)
        #self.classifier = svm.LinearSVC()
        self.classifier = svm.SVC(kernel='linear')
        self.classifier.fit(trainTable,iListLabels)

    def test(self,iListDocuments):
        testTable = self.vsm.test(iListDocuments)
        predictedLabels = self.classifier.predict(testTable)
        return predictedLabels
        
    def evaluation(self,trueLabels,predictedLabels):
        accuracy = metrics.accuracy_score(trueLabels,predictedLabels)
        precision = metrics.precision_score(trueLabels,predictedLabels,pos_label=None)
        recall = metrics.recall_score(trueLabels,predictedLabels,pos_label=None)
        f1 = metrics.f1_score(trueLabels,predictedLabels,pos_label=None)
        return (accuracy,precision,recall,f1)
