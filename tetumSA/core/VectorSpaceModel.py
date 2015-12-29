# Natural Language Toolkit: code_document_classify_fd
import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from curses.ascii import isspace
from scipy.sparse.base import issparse
class VectorSpaceModel:
#return dictVectorizer & tfidfTran & tfidfTran
	def __init(self):
		pass
	@staticmethod
	def createInstance(typeInstance):
		model = 	{"TFIDF": lambda: TFIDF(),
						"BooleanWeighting": lambda: BooleanWeighting()
						}[typeInstance]()
		return model
	
	def train(self,listDocuments):
		pass
	def test(self,listDocuments):
		pass
	def run(self,listTrainDocuments,listTestDocuments):
		''' Input:
				listTrainDocuments, listTestDocuments: list of documents, each document is a list of string.
		'''
		trainTable = self.train(listTrainDocuments)
		testTable = self.test(listTestDocuments)
		return (trainTable,testTable)
	def runCutoff(self,listDocuments,cutoffPosition):
		listTrainDocuments = listDocuments[:cutoffPosition]
		listTestDocuments = listDocuments[cutoffPosition:]
		return self.run(listTrainDocuments, listTestDocuments)
	
	def getFeatureNames(self):
		return self.dictVect.get_feature_names()
	
class TFIDF(VectorSpaceModel):
	def __init__(self):
		self.dictVect = None
		self.tfidfTran = None
	def train(self,listDocuments):
		#print "Create the frequency table"
		# Create the frequency table
		freqTable = []
		for document in listDocuments:
			freqTable.append(nltk.FreqDist(document))
		#print "Create the count table"
		# Create the count table (sparse matrix)
		self.dictVect = DictVectorizer()
		trainCountTable = self.dictVect.fit_transform(freqTable)
		#print "Create TF-IDF table"
		# Create TF-IDF table (sparse matrix)
		self.tfidfTran = TfidfTransformer()
		trainTFIDFTable = self.tfidfTran.fit_transform(trainCountTable)
		return trainTFIDFTable
	def test(self,listDocuments):
		#print "Create the frequency table"
		# Create the frequency table
		freqTable = []
		for document in listDocuments:
			freqTable.append(nltk.FreqDist(document))
		#print "Create the count table"
		# Create the count table (sparse matrix)
		testCountTable = self.dictVect.transform(freqTable)
		#print "Create TF-IDF table"
		# Create TF-IDF table (sparse matrix)
		testTFIDFTable = self.tfidfTran.transform(testCountTable)
		return testTFIDFTable
	def getFeatureNames(self):
		return self.dictVect.get_feature_names()
	def getDocumentFrequency(self):
		''' Output:
				df = a list (a list of integers indicating the number of times the current term appears in all list of documents.)
		'''
		df = [sum(self.dictVect[idx]) for idx in range(len(self.getFeatureNames()))]
		return df

class BooleanWeighting(VectorSpaceModel):
	def __init__(self):
		self.dictVect = None
	def train(self,listDocuments):
		freqTable = []
		for document in listDocuments:
			freqTable.append(nltk.FreqDist(set(document)))
		# "Create the count table"
		# Create the count table (sparse matrix)
		self.dictVect = DictVectorizer()
		trainCountTable = self.dictVect.fit_transform(freqTable)
		return trainCountTable

	def test(self,listDocuments):
		freqTable = []
		for document in listDocuments:
			freqTable.append(nltk.FreqDist(set(document)))
		testCountTable = self.dictVect.transform(freqTable)
		return testCountTable

#----------------------------------------------------------------------
if __name__ == "__main__":
	#listDocuments = ["Apple Stock will increase tomorrow","I want to eat more apple","can i increase more","will will will"]
	listDocuments = [["I","have", "two", "two"],
					["two", "three", "four"],
					["two","three"],
					["I", "have", "three", "three"]]
	vsm = VectorSpaceModel.createInstance("BooleanWeighting")
	(trainTable,testTable)=vsm.runCutoff(listDocuments,2)
	print "Train Table:"
	print trainTable
	print "Test Table:"
	print testTable	
	print vsm.getFeatureNames()
	df = trainTable.tocsc().sum(0)
	listExtractedAspects = vsm.getFeatureNames()
	for idx in reversed(range(df.shape[1])):
		if df[0,idx] < 2:
			del listExtractedAspects[idx]
	print df
	print listExtractedAspects
