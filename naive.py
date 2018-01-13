import numpy as np
from sklearn.model_selection import KFold
import math

categorical = []

def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(0, inpdata.shape[1]-2), dtype = 'S15')
    labels = np.loadtxt(filename,delimiter = '\t', usecols = inpdata.shape[1]-1, dtype = 'S15')
    return X, labels

def calCategoricalProb(train_data, train_label, query_list):
	total = len(train_label)
	count_0 = train_label.tolist().count("0")
	count_1 = train_label.tolist().count("1")
	h_0 = count_0 / total
	h_1 = count_1 / total
	p_x = 1
	for q in query_list:
		q_count = ((train_data == q).sum())/total
		p_x = p_x * q_count
	#counting for label 0
	query_prob_0 = 1
	for i in range(0, len(query_list)):
		c_q_0 = 0
		for j in range(0, train_data.shape[0]):
			if(train_data[j][i] == query_list[i] and train_label[j] == '0'):
				c_q_0 += 1
		query_prob_0 = query_prob_0 * (c_q_0/count_0)
	query_prob_1 = 1
	for i in range(0, len(query_list)):
		c_q_1 = 0
		for j in range(0, train_data.shape[0]):
			if(train_data[j][i] == query_list[i] and train_label[j] == '1'):
				c_q_1 += 1
		query_prob_1 = query_prob_1 * (c_q_1/count_1)
	final_prob_0 = query_prob_0 * (h_0 / p_x)
	final_prob_1 = query_prob_1 * (h_1 / p_x)
	d = {}
	d['0'] = final_prob_0
	d['1'] = final_prob_1
	return d


def calNumericalProb(train_data, test_data):
	train_means = np.mean(train_data, axis = 0)
	train_std = np.std(train_data, axis = 0)
	exponent = np.exp(-(np.power(test_data-train_means,2)/(2*np.power(train_std,2))))
	probability = (1 / (np.sqrt(2*math.pi) * train_std)) * exponent
	return probability


def NaiveBayes(train_data, train_label, test_data):
	#Do calculation for numerical data
	train_numerical_data = np.delete(train_data, categorical, axis = 1)
	test_numerical_data = np.delete(test_data, categorical, axis = 1)
	train_numerical_data = train_numerical_data.astype(np.float)
	test_numerical_data = test_numerical_data.astype(np.float)
	train_numerical_data_0 = train_numerical_data[train_label == '0']  
	train_numerical_data_1 = train_numerical_data[train_label == '1'] 

	numerical_prob_0 = calNumericalProb(train_numerical_data_0, test_numerical_data)
	numerical_prob_1 = calNumericalProb(train_numerical_data_1, test_numerical_data)

	#Do calculation for categorical data

	categorical_prob_0 = []
	categorical_prob_1 = []

	if(len(categorical) != 0):
		train_categorical_data = train_data[:, categorical]
		test_categorical_data = test_data[:, categorical]
		train_categorical_data = train_categorical_data.astype(str)
		test_categorical_data = test_categorical_data.astype(str)
		
		for i in range(0, len(test_categorical_data)):
			prob = calCategoricalProb(train_categorical_data, train_label, [test_categorical_data[i][0]])
			categorical_prob_0.append(prob['0'])
			categorical_prob_1.append(prob['1'])
		categorical_prob_0 = np.array(categorical_prob_0)
		categorical_prob_1 = np.array(categorical_prob_1)
	else:
		for i in range(0, len(test_data)):
			categorical_prob_0.append(1)
			categorical_prob_1.append(1)

	#print(train_categorical_data)
	#print(test_categorical_data)

	test_algo_obtained_labels = []
	for i in range(0, len(test_data)):
		prob_0 = np.prod(numerical_prob_0[i]) * categorical_prob_0[i]
		prob_1 = np.prod(numerical_prob_1[i]) * categorical_prob_1[i]
		if(prob_0 > prob_1):
			test_algo_obtained_labels.append("0")
		else:
			test_algo_obtained_labels.append("1")
	return test_algo_obtained_labels


def evalPerformance(test_algo_obtained_labels, test_labels):
	a, b, c, d = 0, 0, 0, 0
	test_labels = np.array(test_labels)
	test_algo_obtained_labels = np.array(test_algo_obtained_labels)
	test_labels = test_labels.astype(np.float)
	test_algo_obtained_labels = test_algo_obtained_labels.astype(np.float)
	for i in range(0, len(test_labels)):
		if(test_labels[i] == test_algo_obtained_labels[i] and test_labels[i] == 1.0):
			a += 1
		elif(test_labels[i] == 1.0 and test_algo_obtained_labels[i] == 0.0):
			b += 1
		elif(test_labels[i] == 0.0 and test_algo_obtained_labels[i] == 1.0):
			c += 1
		elif(test_labels[i] == test_algo_obtained_labels[i] and test_labels[i] == 0.0):
			d += 1
	accuracy = float(a+d)/(a+b+c+d)
	precision = a / float(a+c)
	recall = a/float(a+b)
	f_measure = 2*a/float(2*a + b + c)
	print(" Accuracy is : " + str(accuracy)),
	print(" Precision is : " + str(precision))
	print(" Recall is : " + str(recall))
	print(" F Measure is : " + str(f_measure))
	print()
	return accuracy, precision, recall, f_measure

def KFoldValidation(data, labels, n_s):
	kf = KFold(n_splits=n_s)
	accuracy_list = []
	precision_list = []
	recall_list = []
	f_measure_list = []
	for train, test in kf.split(data):
	    train_data, test_data = np.array(data)[train], np.array(data)[test]
	    train_label, test_label = np.array(labels)[train], np.array(labels)[test]
	    test_algo_obtained_labels = NaiveBayes(train_data, train_label, test_data)
	    accuracy, precision, recall, f_measure  = evalPerformance(test_algo_obtained_labels, test_label)
	    accuracy_list.append(accuracy)
	    precision_list.append(precision)
	    recall_list.append(recall)
	    f_measure_list.append(f_measure)
	avg_accuracy = sum(accuracy_list)/len(accuracy_list)
	avg_precision = sum(precision_list)/len(precision_list)
	avg_recall = sum(recall_list)/len(recall_list)
	avg_f_measure = sum(f_measure_list)/len(f_measure_list)

	print(" Average Accuracy is : " + str(avg_accuracy))
	print(" Average Precision is : " + str(avg_precision))
	print(" Average Recall is : " + str(avg_recall))
	print(" Average F Measure is : " + str(avg_f_measure))
	print()
	return 0


X, labels = preprocess("project3_dataset1.txt")

for i in range(0, X.shape[1]):
	try:
		X[:, i] = X[:, i].astype(np.float)
	except:
		'''
		b, c = np.unique(X[:,4], return_inverse=True)
		X[:,i] = c
		X[:, i] = X[:, i].astype(np.float)
		'''
		categorical.append(i)


labels = labels.astype(str)
KFoldValidation(X, labels, 10)

