import csv
from csv import reader
from math import log
from collections import defaultdict, Counter

"""
* ECON1660
* PS4: Trees
*
* Fill in the functions that are labaled "TODO".  Once you
* have done so, uncomment (and adjust as needed) the main
* function and the call to main to print out the tree and
* the classification accuracy.
"""


"""
* TODO: Create features to be used in your regression tree.
"""

"""************************************************************************
* function: partition_loss(subsets)
* arguments:
* 		-subsets:  a list of lists of labeled data (representing groups
				   of observations formed by a split)
* return value:  loss value of a partition into the given subsets
*
* TODO: Write a function that computes the loss of a partition for
*       given subsets
************************************************************************"""
def partition_loss(subsets):
	
	mse = 0
	net_count = 0

	for subset in subsets:
		net_count += len(subset)

		avg = 0
		for el in subset:
			avg += el[1]

		avg /= len(subset)

		for el in subset:
			mse += (el[1] - avg) * (el[1] - avg)

	return mse / net_count




"""************************************************************************
* function: partition_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  a list of lists, where each list represents a subset of
*				 the inputs that share a common value of the given 
*				 attribute
************************************************************************"""
def partition_by(inputs, attribute):
	groups = defaultdict(list)
	for input in inputs:
		key = input[0][attribute]	#gets the value of the specified attribute
		groups[key].append(input)	#add the input to the appropriate group
	return groups


"""************************************************************************
* function: partition_loss_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  the loss value of splitting the inputs based on the
*				 given attribute
************************************************************************"""
def partition_loss_by(inputs, attribute):
	partitions = partition_by(inputs, attribute)
	return partition_loss(partitions.values())


"""************************************************************************
* function:  build_tree(inputs, num_levels, split_candidates = None)
*
* arguments:
* 		-inputs:  labeled data used to construct the tree; should be in the
*				  form of a list of tuples (a, b) where 'a' is a dictionary
*				  of features and 'b' is a label
*		-num_levels:  the goal number of levels for our output tree
*		-split_candidates:  variables that we could possibly split on.  For
*							our first level, all variables are candidates
*							(see first two lines in the function).
*			
* return value:  a tree in the form of a tuple (a, b) where 'a' is the
*				 variable to split on and 'b' is a dictionary representing
*				 the outcome class/outcome for each value of 'a'.
* 
* TODO:  Write a recursive function that builds a tree of the specified
*        number of levels based on labeled data "inputs"
************************************************************************"""
def build_tree(inputs, num_levels, split_candidates = None):
	#if first pass, all keys are split candidates
	if split_candidates == None:
		split_candidates = list(inputs[0][0].keys())

	if len(split_candidates) == 0 or num_levels == 0:
		days_until_funded_sum = 0
		for input_ in inputs:
			days_until_funded_sum += input_[1]
		return days_until_funded_sum / len(inputs)

	minAttr = ""
	minVal = float('inf')

	for candidate in split_candidates:
		curr = partition_loss_by(inputs, candidate)
		if curr < minVal:
			minVal = curr
			minAttr = candidate
	
	partition = partition_by(inputs, minAttr)

	split_candidates.remove(minAttr)
	dicRet = {}
	for k, v in partition.items():
		dicRet[k] = build_tree(v, num_levels - 1, split_candidates)
	return (minAttr, dicRet)


"""************************************************************************
* function:  classify(tree, to_classify)
*
* arguments:
* 		-tree:  a tree built with the build_tree function
*		-to_classify:  a dictionary of features
*
* return value:  a value indicating a prediction of days_until_funded

* TODO:  Write a recursive function that uses "tree" and the values in the
*		 dictionary "to_classify" to output a predicted value.
************************************************************************"""
def classify(tree, to_classify):
	attribute = tree[0]
	node = tree[1][to_classify[attribute]]
	if type(node) is tuple:
		return classify(node, to_classify)
	return node


"""************************************************************************
* function:  load_data()
* arguments:  N/A
* return value:  a list of tuples representing the loans data
* 
* TODO:  Read in the loans data from the provided csv file.  Store the
* 		 observations as a list of tuples (a, b), where 'a' is a dictionary
*		 of features and 'b' is the value of the days_until_funded variable
************************************************************************"""
def load_data():
	with open('tables/loans_A_labeled.csv', 'r') as read_obj:
		loans = []
		# pass the file object to reader() to get the reader object
		csv_reader = reader(read_obj)
		# Iterate over each row in the csv using reader object
		labels = None
		for i, row in enumerate(csv_reader):
			# row variable is a list that represents a row in csv
			if i == 0:
				labels = row
			else:
				features = {}
				days_until_funded = -1
				for j, el in enumerate(row):
					if j == len(row) - 1:
						days_until_funded = int(el)
					else:
						features[labels[j]] = el
				loans.append((features, days_until_funded))

		return loans
# load_data()
print(load_data())