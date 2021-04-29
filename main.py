
import os
import argparse
import gc
import random
from inout import get_data, print_stats
#from svm_wrapper import SVMLiblinear
from sklearn.metrics import f1_score
from featureselection.igi import IGI

"""
from classifiers.config import getClassifierInfo
from classifiers.traditionalClassifiers import TraditionalClassifier
"""


def arguments():
	parser = argparse.ArgumentParser(description='Generate baseline splits.')
	parser.add_argument('-d','--dataset', type=str, default="aisopos_ntua_2L")
	parser.add_argument('--folds', type=int, default=10)
	#parser.add_argument('--param', type=str, default="c-5,13,3-e0.001-n12")

	args = parser.parse_args()

	args.inputdir = f'datasets/{args.dataset}/tfidf/'
	args.outputdir = f'out/{args.dataset}/'

	print(args)
	
	#if not os.path.exists(args.outputdir):
	#	os.makedirs(args.outputdir, exist_ok=True)

	return args

def main():
	gc.collect()

	args = arguments()
	random.seed(1608637542)

	#info = getClassifierInfo("svm")

	micro_list = []
	macro_list = []

	for f in range(args.folds):
	#for f in range(1):
		print("Fold {}".format(f))
		X_train, y_train, X_test, y_test, n_classes, initial_vocab = get_data(args.inputdir, f)


		#igi = IGI(n_features=1000)
		igi = IGI(n_features=200)
		igi.fit(X_train, y_train)

		#igi.print_most_important_words(initial_vocab)

		X_train = igi.transform(X_train)
		X_test  = igi.transform(X_test)

		"""
		classifier = TraditionalClassifier(info)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)

		micro = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
		macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
		print("F1-Score")
		print("\tMicro: ", micro)
		print("\tMacro: ", macro)

		micro_list.append(micro)
		macro_list.append(macro)
		"""

	#med_mic,error_mic,med_mac,error_mac = print_stats(micro_list, macro_list)



if __name__ == '__main__':
	main()
