import csv
import datetime
import random
from csv import reader
from math import log
from fractions import Fraction
from collections import defaultdict, Counter
from random import choices
from random import seed
from random import random

"""
* ECON1660
* PS4: Trees
*
* Fill in the functions that are labaled "TODO".  Once you
* have done so, uncomment (and adjust as needed) the main
* function and the call to main to print out the tree and
* the classification accuracy.
"""

TRAIN_SPLIT_VAL = 0.8

model1_attr = ["loan_amount_bin", "repayment_term_bin"]
dateTimeModel_attr = ["holiday_time", "day_of_week", "waking_hours"]
countryModel_attr = ["country_region", "country_sub_region", "gdp_small_bin"]
model2_attr = ["age_bin", "gender", "pictured", "pop_name"]
model3_attr = ["long", "fam", "smart", "sympathy"]
model_basic = ["gender", "pictured", "sector", "country", "languages"]

models = [model1_attr, model2_attr, dateTimeModel_attr, model3_attr, countryModel_attr]
forest_base_model = model1_attr + model2_attr + model3_attr + dateTimeModel_attr + countryModel_attr + model_basic


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
def build_tree(inputs, num_levels, model_num, split_candidates = None):
	#if first pass, all keys are split candidates
	if split_candidates == None:
		# split_candidates = models[model_num]
		split_candidates = models[model_num] + model_basic

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

	split_candidates_ = split_candidates.copy()
	split_candidates_.remove(minAttr)
	dicRet = {}
	for k, v in partition.items():
		dicRet[k] = build_tree(v, num_levels - 1, 0, split_candidates_)
	return (minAttr, dicRet)


def build_forest_tree(inputs, num_levels, num_split_candidates, split_candidates=None):
	#if first pass, all keys are split candidates
	if split_candidates == None:
		split_candidates = forest_base_model

	if len(split_candidates) == 0 or num_levels == 0:
		days_until_funded_sum = 0
		for input_ in inputs:
			days_until_funded_sum += input_[1]
		return days_until_funded_sum / len(inputs)

	minAttr = ""
	minVal = float('inf')

	sampled_split_candidates = None
	if len(split_candidates) <= num_split_candidates:
		sampled_split_candidates = split_candidates
	else:
		sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

	for candidate in sampled_split_candidates:
		curr = partition_loss_by(inputs, candidate)
		if curr < minVal:
			minVal = curr
			minAttr = candidate
	
	partition = partition_by(inputs, minAttr)

	split_candidates_ = split_candidates.copy()
	split_candidates_.remove(minAttr)
	dicRet = {}
	for k, v in partition.items():
		dicRet[k] = build_tree(v, num_levels - 1, 0, split_candidates_)
	return (minAttr, dicRet)


def forest_predict(trees, to_classify):
	predictions = []
	for tree in trees:
		predictions.append(classify(tree, to_classify))
	return sum(predictions) / len(predictions)
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
	if to_classify[attribute] not in tree[1]:
		print("BAD")
		return 7
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
def load_data(filename):
	with open(filename, 'r') as read_obj:
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
					if filename != "tables/loans_B_unlabeled.csv" and j == len(row) - 1:
						days_until_funded = int(el)
					else:
						features[labels[j]] = el
				loans.append((features, days_until_funded))

		return loans


def splitData(data):
	seed()
	test = []
	train = []
	for point in data:
		if random() > TRAIN_SPLIT_VAL:
			test.append(point)
		else:
			train.append(point)

	return test, train


def test_model(tree, data):
	mse = 0
	for point in data:
		prediction = classify(tree, point[0])
		mse += (point[1] - prediction) * (point[1] - prediction)
	return mse / len(data)


def test_forest(forest, data):
	mse = 0
	for point in data:
		prediction = forest_predict(forest, point)
		mse += (point[1] - prediction) * (point[1] - prediction)
	return mse / len(data)


def write_predictions(model, data):
	with open("loans_B_predicted_JB_AL_JS", 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=['ID', 'days_until_funded_JB_AL_JS'])
		writer.writeheader()
		for point in data:
			prediction = classify(model, point)
			writer.writerow({'ID': point[0]['id'], 'days_until_funded_JB_AL_JS': prediction})



def make_model1_data(loans):
	# find mean loan_amount and mean repayment_amount
	loan_amount_tot = 0
	repayment_term_tot = 0
	for loan in loans:
		loan_amount_tot += int(loan[0]["loan_amount"])
		repayment_term_tot += int(loan[0]["repayment_term"])

	count = len(loans)
	mean_loan_amount = loan_amount_tot / count
	mean_repayment_term = repayment_term_tot / count

	for loan in loans:
		curr_loan_amount = int(loan[0]["loan_amount"])
		curr_repayment_term = int(loan[0]["repayment_term"])
		if curr_loan_amount <= mean_loan_amount:
			loan[0]["loan_amount_bin"] = 0
		else:
			loan[0]["loan_amount_bin"] = 1

		if curr_repayment_term <= mean_repayment_term:
			loan[0]["repayment_term_bin"] = 0
		else:
			loan[0]["repayment_term_bin"] = 1

	return loans

def make_model3_data(loans):
	for loan in loans:
		fam_score = 0
		fam_words = ["children", "child", "married", "kids", "family"]
		smart_score = 0
		smart_words = ["entrepreneur", "entrepreneurship", "business"]
		sympathy_score = 0
		sympathy_words = ["rural", "necessary", "community", "single", "dependent"]
		description = loan[0]["description"]
		des_len = len(description)

		if des_len > 1000:
			loan[0]["long"] = 2
		elif des_len < 400:
			loan[0]["long"] = 0
		else:
			loan[0]["long"] = 1

		for f_word in fam_words:
			fam_score += description.count(f_word)
		if fam_score > 0:
			loan[0]["fam"] = 1
		else:
			loan[0]["fam"] = 0

		for sm_words in smart_words:
			smart_score += description.count(sm_words)
		if smart_score > 0:
			loan[0]["smart"] = 1
		else:
			loan[0]["smart"] = 0

		for sy_word in sympathy_words:
			sympathy_score += description.count(sy_word)
		if sympathy_score > 0:
			loan[0]["sympathy"] = 1
		else:
			loan[0]["sympathy"] = 0
	return loans

def make_dateTimeModel_data(loans):
	for loan in loans:
		loan_data = loan[0]
		posted_date = loan_data["posted_date"]
		dateTime = datetime.datetime.strptime(posted_date, '%Y-%m-%dT%H:%M:%SZ')

		loan_data["day_of_week"] = dateTime.weekday()

		month = dateTime.month
		day = dateTime.day
		if (month == 12 and day > 15) or (month == 1 and day < 15):
			loan_data["holiday_time"] = 1
		else:
			loan_data["holiday_time"] = 0

		hour = dateTime.hour
		if hour >= 9 and hour <= 17:
			loan_data["waking_hours"] = 1
		else:
			loan_data["waking_hours"] = 0
	return loans

def make_gdp_data(loans):
	small_gdp = []
	with open('tables/smaller_gdp.csv', 'r') as read_obj:
		csv_reader = reader(read_obj)
		for i, row in enumerate(csv_reader):
			if i > 0:
				small_gdp.append(row[0])

	country_regions = {}
	country_sub_regions = {}
	with open('tables/country_regions.csv', 'r') as read_obj:
		csv_reader = reader(read_obj)
		for i, row in enumerate(csv_reader):
			if i > 0:
				country_regions[row[0]] = row[5]
				country_sub_regions[row[0]] = row[6]


	for loan in loans:
		loan_data = loan[0]
		country  = loan_data["country"]
		loan_data["country_region"] = country_regions[country]
		loan_data["country_sub_region"] = country_sub_regions[country]

		if country in small_gdp:
			loan_data["gdp_small_bin"] = 0
		else:
			loan_data["gdp_small_bin"] = 1


	return loans


def make_model2_data(loans):
	for loan in loans:
		age = findAge(loan[0]["description"])
		if age < 0:
			loan[0]["age_bin"] = "unknown"
		elif age < 30:
			loan[0]["age_bin"] = "young"
		elif age < 50:
			loan[0]["age_bin"] = "adult"
		else:
			loan[0]["age_bin"] = "old"

	names = set()
	with open('baby-names.csv', 'r') as read_obj:
		csv_reader = reader(read_obj)
		for i, row in enumerate(csv_reader):
			if i > 0 and int(row[0]) > 1980:
				names.add(row[1])

	for loan in loans:
		if loan[0]["name"] in names:
			loan[0]["pop_name"] = 1
		else:
			loan[0]["pop_name"] = 0

	return loans


def findAge(description):
	for i in range(len(description) - 1):
		st = description[i:i+2]
		if st.isnumeric():
			try:
				ints = int(description[i:i+2])
				return ints
			except:
				return -1
	return -1


def make_model_data(data, model_num):
	if model_num == 0:
		return make_model1_data(data)
	elif model_num == 1:
		return make_model2_data(data)
	elif model_num == 2:
		return make_dateTimeModel_data(data)
	elif model_num == 3:
		return make_model3_data(data)
	elif model_num == 4:
		return make_gdp_data(data)

def make_all_model_data(data):
	for i in range(len(models)):
		data = make_model_data(data, i)
	return data

def bootstrap(input_loans, n):
	return choices(input_loans, k = n)

def test_train_split(model_num):
	test, train = splitData(make_model_data(load_data("tables/loans_AB_labeled.csv"), model_num))
	tree = build_tree(train, len(models[model_num]) + len(model_basic), model_num)
	print(test_model(tree, test))


print(test_train_split(1))


def bootstrap(input_loans, n):
	return choices(input_loans, k = n)


def test_train_split_forest(model_num, num_trees, num_levels, num_split_candidates, n):
	test, train = splitData(make_all_model_data(load_data("tables/loans_AB_labeled.csv")))
	forest = []
	for _ in range(num_trees):
		train_sample = bootstrap(train, n)
		forest.append(build_forest_tree(train_sample, num_levels, num_split_candidates))
	print(test_forest(forest, test))
