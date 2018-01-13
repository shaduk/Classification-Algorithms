import numpy as np
import plotly
from plotly.graph_objs import *
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from heapq import heappush, heappop
import pprint
import random


def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(0, inpdata.shape[1]-1), dtype = 'S15')
    labels = np.loadtxt(filename,delimiter = '\t', usecols = inpdata.shape[1]-1, dtype = 'S15')
    return X, labels

def calculate_gini(train_data, splitIndex, j, labels, nodeIndices):
	node1_class1 = 0.0
	node1_class2 = 0.0
	node2_class1 = 0.0
	node2_class2 = 0.0
	total_1 = 0
	total_2 = 0
	#print("splits on ", str(train_data[splitIndex,j]))
	for index in range(0, len(train_data[:,j])):
		if index in nodeIndices:
			if(train_data[index, j] < train_data[splitIndex,j]):
				if(labels[index] == 0):
					node1_class1 += 1
				else:
					node1_class2 += 1
				total_1 += 1
			else:
				if(labels[index] == 0):
					node2_class1 += 1
				else:
					node2_class2 += 1
				total_2 += 1
	gini = 1.0
	if(total_1 != 0 and total_2 != 0):
		eq_1 = (1 - (node1_class1/float(total_1))**2 - (node1_class2 / float(total_1))**2)
		eq_2 = (1 - (node2_class1/float(total_2))**2 - (node2_class2 / float(total_2))**2)
		gini = (total_1/float(total_1 + total_2)) * eq_1 + (total_2/float(total_1 + total_2))*eq_2

	return gini

def evalPerformance(test_algo_obtained_labels, test_labels):
	a, b, c, d = 0, 0, 0, 0
	test_labels = np.array(test_labels)
	test_algo_obtained_labels = np.array(test_algo_obtained_labels)
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
	accuracy = 0.0
	precision = 0.0
	recall = 0.0
	f_measure = 0.0
	if(a+b+c+d != 0):
		accuracy = float(a+d)/(a+b+c+d)
	if(a+c != 0):
		precision = a / float(a+c)
	if(a+b != 0):
		recall = a/float(a+b)
	if(2*a + b + c != 0):
		f_measure = 2*a/float(2*a + b + c)
	print(" Accuracy is : " + str(accuracy)),
	print(" Precision is : " + str(precision))
	print(" Recall is : " + str(recall))
	print(" F Measure is : " + str(f_measure))
	print()
	return accuracy, precision, recall, f_measure

def checkTerminal(indices, labels):
	if(len(indices) < 1):
		return -1
	last = labels[indices[0]]
	for i in range(1, len(indices)):
		if(labels[indices[i]] != last):
			return -1
		last = labels[indices[i]]
	return last

def get_split(nodeIndices, train_data, labels, depth):
	if(len(nodeIndices) == 0):
		return
	gini_value = 1.0
	splitsOn = 0
	splitOnIndex = 0
	featureIndex = 0
	for feature in range(0, train_data.shape[1]):
		already_checked = {}
		for i in range(0, len(train_data)):
			if(i in nodeIndices):
				if train_data[i, feature] not in already_checked:
					gini_temp = calculate_gini(train_data, i, feature, labels, nodeIndices)
					already_checked[train_data[i, feature]] = 1
					if(gini_temp < gini_value):
						splitsOn = train_data[i, feature]
						splitOnIndex = i
						gini_value = gini_temp
						featureIndex = feature
	node = {}
	node["value"] = float(splitsOn)
	node["featureIndex"] = featureIndex
	leftIndices = []
	rightIndices = []
	leftlabels = []
	rightlabels = []
	for i in range(0, len(train_data)):
		if(i in nodeIndices):
			if(train_data[i, featureIndex] < splitsOn):
				leftIndices.append(i)
				leftlabels.append(labels[i])
			else:
				rightIndices.append(i)
				rightlabels.append(labels[i])

	node["leftLeaf"] = checkTerminal(leftIndices, labels)

	if(node["leftLeaf"] == -1):
		# if(len(leftlabels) < 0.05*train_data.shape[0]):
		# 	node["leftLeaf"] = most_common(leftlabels)
		# else:
		if(len(leftIndices) == 0):
			node["leftLeaf"] = labels[splitOnIndex]
		elif(depth == 0):
			node["leftLeaf"] = most_common(leftlabels)
		else:
			node["leftLeaf"] = get_split(leftIndices, train_data, labels, depth - 1)

	node["rightLeaf"] = checkTerminal(rightIndices, labels)
	if(node["rightLeaf"] == -1):
		# if(len(rightlabels) < 0.05*train_data.shape[0]):
		# 	node["rightLeaf"] = most_common(rightlabels)
		# else:
		if(len(rightIndices) == 0):
			node["rightLeaf"] = labels[splitOnIndex]
		elif(depth == 0):
			node["rightLeaf"] = most_common(rightlabels)
		else:
			node["rightLeaf"] = get_split(rightIndices, train_data, labels, depth - 1)
	return node

def traverseTree(root, test_row):
	#print(test_row)
	if(root == None):
		return -1
	#print(test_row)
	featureIndex = root["featureIndex"]
	#print(featureIndex)
	if(test_row[featureIndex] < root["value"]):
		if(root["leftLeaf"] != 0.0 and root["leftLeaf"] != 1.0):
			return traverseTree(root["leftLeaf"], test_row)
		else:
			return root["leftLeaf"]
	else:
		if(root["rightLeaf"] != 0.0 and root["rightLeaf"] != 1.0):
			return traverseTree(root["rightLeaf"], test_row)
		else:
			return root["rightLeaf"]
	return -1

def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

def random_sample(train_data, train_label, weights):
	sample_data = []
	sample_labels = []
	data_size = len(train_data) 
	indexes = []
	weights = normalize(weights)
	while(len(sample_data) < data_size):
		index = np.random.choice(np.arange(0, len(train_data)), p= weights)
		indexes.append(index)
		sample_data.append(train_data[index, :])
		sample_labels.append(train_label[index])
	return np.array(sample_data), np.array(sample_labels)


def most_common(lst):
    return max(set(lst), key=lst.count)	

def Boosting(train_data, train_label, test_data, rounds, depth):
	nodeIndices = range(len(train_data))
	weights = [1/float(train_data.shape[0]) for i in range(train_data.shape[0])]
	classifier_error = 0
	classifier_importance = []
	trees = []
	test_labels = []
	n = 0
	while(n < rounds):
		featureIndex_list = []
		classifier_imp_variable = 0
		sample_data, sample_labels = random_sample(train_data, train_label, weights)
		treeHead = get_split(nodeIndices, sample_data, sample_labels, depth)
		train_algo_labels = []
		for row in train_data:
			label = traverseTree(treeHead, row)
			train_algo_labels.append(label)
		numerator = 0
		sum_of_weights = 0
		for i in range(0, len(train_algo_labels)):
			if(train_label[i] != train_algo_labels[i]):
				numerator += weights[i] * 1
			else:
				numerator += weights[i] * 0
			sum_of_weights += weights[i]
		classifier_error = numerator/float(sum_of_weights)
		classifier_imp_variable = 0.5*np.log((1-classifier_error)/float(classifier_error))
		numerator = 0
		for i in range(0, len(weights)):
			if(train_label[i] != train_algo_labels[i]):
				numerator = np.exp(-1*(-1)*classifier_imp_variable)
			else:
				numerator = np.exp(-1*1*classifier_imp_variable)
			weights[i] = (weights[i] * numerator)/sum(weights)
		if(classifier_error < 0.5):
			classifier_importance.append(classifier_imp_variable)
			trees.append(treeHead)
			n += 1

	for row in test_data:
		sumvalue = 0
		for i in range(rounds):
			label = traverseTree(trees[i], row)
			if label == 0.0:
				value = -1*classifier_importance[i]
			else:
				value = classifier_importance[i]
			sumvalue += value
		if(sumvalue < 0):
			test_labels.append(0)
		else:
			test_labels.append(1)
	return test_labels
	

def KFoldValidation(data, labels, n_s):
	kf = KFold(n_splits=n_s)
	accuracy_list = []
	precision_list = []
	recall_list = []
	f_measure_list = []
	for i in range(0, data.shape[1]):
		try:
			data[:, i] = data[:, i].astype(np.float)
		except:
			b, c = np.unique(X[:,4], return_inverse=True)
			data[:,i] = c
			data[:, i] = data[:, i].astype(np.float)
	data = data.astype(np.float)
	labels = labels.astype(np.float)
	for train, test in kf.split(data):
	    train_data, test_data = np.array(data)[train], np.array(data)[test]
	    train_label, test_label = np.array(labels)[train], np.array(labels)[test]
	    test_algo_obtained_labels = Boosting(train_data, train_label, test_data, 10, 3)
	    same = 0
	    diff = 0
	    for i in range(0, len(test_algo_obtained_labels)):
	    	if(test_algo_obtained_labels[i] == test_label[i]):
	    		same += 1
	    	else:
	    		diff += 1
	    print("accuracy : ", str(same/float(same + diff)))

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
KFoldValidation(X, labels, 10)
