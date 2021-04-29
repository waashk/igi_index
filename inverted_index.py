

#based on code: https://www.kaggle.com/donkeys/tf-idf-and-inverted-index-creation-for-covid19

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def features_for_doc(X: TfidfVectorizer, idx: int):
	features_indexes = X[idx , :].nonzero()[1]
	return features_indexes

def get_inverted_index_from_tfidf(X: TfidfVectorizer):

	inverted_index = defaultdict(list)

	for idx in tqdm(range(X.shape[0]),desc="Setting Inverted index"):
	#for idx in range(1):
		features_indexes = features_for_doc(X, idx)
		for f in features_indexes:
			inverted_index[f].append(idx)

	return inverted_index
	#print(inverted_index)