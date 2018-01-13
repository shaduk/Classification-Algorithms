import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from heapq import heappush, heappop

categorical = []

def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(0, inpdata.shape[1]-1), dtype = 'S15')
    labels = np.loadtxt(filename,delimiter = '\t', usecols = inpdata.shape[1]-1, dtype = 'S15')
    return X, labels


def distance(x, m):
	global categorical
	dis = 0
	for i in range(len(x)):
		if(i not in categorical):
			dis = dis + np.square(np.subtract(float(x[i]), float(m[i])))
		else:
			if(x[i] != m[i]):
				dis += 1
	#print("distance between "+str(x)+" and "+str(m) + " is " + str(dis))
	return np.sqrt(dis)

def kNeighbours(train_data, test_data, pointIndex, k):
	heap = []
	k_neighbours = []
	for i in range(len(train_data)):
		dist = distance(train_data[i], test_data[pointIndex])
		heappush(heap, (dist, i))
	for i in range(k):
		k_neighbours.append(heappop(heap))
	return k_neighbours
	
def most_common(lst):
    return max(set(lst), key=lst.count)	

def kNN(train_data, train_label, test_data, k):
	test_labels = []
	for pointIndex in range(0, len(test_data)):
		k_neighbours = kNeighbours(train_data, test_data, pointIndex, k)
		labels = []
		#print("for point :", str(test_data[pointIndex]))
		#print(k_neighbours)
		for neighbor in k_neighbours:
			labels.append(train_label[neighbor[1]])
		label = most_common(labels)
		test_labels.append(label)
	#print("labels are ")
	#print(test_labels)
	return test_labels

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

def KFoldValidation(data, labels, n_s, K):
	kf = KFold(n_splits=n_s)
	accuracy_list = []
	precision_list = []
	recall_list = []
	f_measure_list = []
	for train, test in kf.split(data):
	    train_data, test_data = np.array(data)[train], np.array(data)[test]
	    train_label, test_label = np.array(labels)[train], np.array(labels)[test]
	    test_algo_obtained_labels = kNN(train_data, train_label, test_data, K)
	    same = 0
	    diff = 0
	    for i in range(0, len(test_algo_obtained_labels)):
	    	if(test_algo_obtained_labels[i] == test_label[i]):
	    		same += 1
	    	else:
	    		diff += 1
	    #print("accuracy : ", str(same/float(same + diff)))
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

'''
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
test_scaled  = scaler.fit_transform(test_data)
print(X_scaled)
print(test_scaled)
print()
kNN(X_scaled, label, test_scaled, 1)
'''

for i in range(0, X.shape[1]):
	try:
		X[:, i] = X[:, i].astype(np.float)
	except:
		b, c = np.unique(X[:,i], return_inverse=True)
		X[:,i] = c
		X[:, i] = X[:, i].astype(np.float)
		categorical.append(i)

labels = labels.astype(np.float)
X = X.astype(np.float)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
KFoldValidation(X_scaled, labels, 10, 5)
#print(LA.norm(X[0]-X[1]))



