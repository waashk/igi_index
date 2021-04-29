
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
from scipy.stats import t as qt

def get_data(inputdir, f):

	X_train, y_train = load_svmlight_file(
		inputdir+"train"+str(f)+".gz", dtype=np.float64)
	X_test, y_test = load_svmlight_file(
		inputdir+"test"+str(f)+".gz", dtype=np.float64)

	# Same vector size
	if (X_train.shape[1] > X_test.shape[1]):
		X_test, y_test = load_svmlight_file(
			inputdir+"test"+str(f)+".gz", dtype=np.float64, n_features=X_train.shape[1])
	elif (X_train.shape[1] < X_test.shape[1]):
		X_train, y_train = load_svmlight_file(
			inputdir+"train"+str(f)+".gz", dtype=np.float64, n_features=X_test.shape[1])

	n_classes = int(max(np.max(y_train), np.max(y_test)))+1

	with open(f"{inputdir}/vocab{f}.csv", 'r') as arq:
		vocab = list(map(str.rstrip, arq.readlines()))

	return X_train, y_train, X_test, y_test, n_classes, vocab

def dump_svmlight_file_txt(X, y, output_dir, set_type="train"):
	valid_sets = ["train", "test"]

	if set_type not in valid_sets:
		raise ValueError(f"Invalid set type. Expected one of: {set_type}")

	filout=f"{output_dir}/{set_type}"

	dump_svmlight_file(X, y, filout, zero_based=False)


def print_stats(micro_list, macro_list):
	#print(micro_list)
	folds = len(micro_list)
	med_mic = np.mean(micro_list)*100
	error_mic = abs(qt.isf(0.975, df=(folds-1))) * \
		np.std(micro_list, ddof=1)/np.sqrt(len(micro_list))*100
	med_mac = np.mean(macro_list)*100
	error_mac = abs(qt.isf(0.975, df=(folds-1))) * \
		np.std(macro_list, ddof=1)/np.sqrt(len(macro_list))*100
	print("Micro\tMacro")
	print("{:.2f}({:.2f})\t{:.2f}({:.2f})".format(
		med_mic, error_mic, med_mac, error_mac))
	return med_mic, error_mic, med_mac, error_mac