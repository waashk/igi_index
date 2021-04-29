
from inverted_index import get_inverted_index_from_tfidf
from collections import defaultdict
import numpy as np
from tqdm import tqdm
class IGI(object):
	"""IGI
	Implementation of IGI method from paper: 
	A new feature selection metric for text classifcation: eliminating the need for a separate pruning stage
	https://link.springer.com/content/pdf/10.1007/s13042-021-01324-6.pdf
	"""
	def __init__(self, n_features: int):
		self.n_features = n_features

	def get_gini_index(self, term_idx, positive_class):

		#print(self.y[self.inverted_index[term_idx]])

		docs_classes = self.y[self.inverted_index[term_idx]]

		tp = np.sum(docs_classes == positive_class)
		fp = np.sum(docs_classes != positive_class)
		fn = self.total_documents - tp
		tn = self.total_documents - fp
		#print(tp, fp, fn, tn)

		tpr = tp / (tp + fn)
		fpr = fp / (tn + fp)
		#print(tpr, fpr)

		igi = ((tpr - fpr)*tpr**2)*((tp / (tp+fp))**2) + ((fpr - tpr)*fpr**2)*(fp / (tp + fp))**2
		
		return igi

	def fit(self, X, y):

		if X.shape[1] <= self.n_features:
			raise("The number of features to be selected must be minor than initial number of features")

		self.y = y
		self.total_documents = len(y)
		self.classes = list(sorted(list(set(y))))
		self.n_classes = len(self.classes)
		#print(self.total_documents)

		#Setting inverted index for optmize calculations
		self.inverted_index = get_inverted_index_from_tfidf(X)
		#Sorted list of features ids
		self.features_idxs = list(sorted(self.inverted_index))
		#Setting if we will compute gini index for feature
		#rules 1. feature present in more than 3 documents
		#rules 2. feature present in less than 25% documents
		self.compute = np.zeros(max(self.features_idxs)+1, dtype=int)
		percent = 0.25*self.total_documents
		for idx in self.features_idxs: 
			if (len(self.inverted_index[idx]) > 3 and len(self.inverted_index[idx]) < percent):
				self.compute[idx] = 1
			else: 
				self.compute[idx] = 0
		#setting gini_index 
		self.gini_index = np.zeros(max(self.features_idxs)+1)


		#setting gini_index not computed features as global minimum
		for idx in self.features_idxs:
			if not self.compute[idx]:
				self.gini_index[idx] = -1000.0


		
		for actual_positive_class in tqdm(self.classes,desc="Computing Gini: "):
			for idx in tqdm(self.features_idxs, "Feature: "):
				if self.compute[idx]:
					self.gini_index[idx] += self.get_gini_index(idx, actual_positive_class)
		
			#print(self.get_gini_index(0, actual_positive_class))
			#print(self.gini_index)
			#print(max(self.gini_index))

			if self.n_classes == 2:
				break


		self.selected_indexes = np.argsort(self.gini_index)[::-1]
		self.selected_indexes = self.selected_indexes[:self.n_features]
		#print(self.selected_indexes)


	def transform(self, X):
		return X[:,sorted(self.selected_indexes)]
		#return X[:,self.selected_indexes]


	def print_most_important_words(self, initial_vocab):
		for idx in self.selected_indexes:
			print(initial_vocab[idx], self.gini_index[idx])