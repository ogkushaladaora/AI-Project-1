#There is a core problem with this code in that it cannot use strings in the generation of the decision tree classifier.
#That would require removing the string stripping lines, and using onehotencoder to convert strings to... something. And then
#promoting whatever that is back to a string in case_3_1. While we're on the subject of case_3_1, I have no idea what I was
#thinking when I wrote that set of if statements to output a decision. 0 is generally not true.

#Anyway if you want to use this, probably don't use the user input option becuase it is hot garbage.

#ogkushaladaora, 2018-11-20

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import misc
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import graphviz
import pydotplus
import io
import collections

from sklearn.tree import export_graphviz

def savetree(tree, features, path):
	f = io.StringIO()
	export_graphviz(tree, out_file = f, feature_names = features)
	pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)

def case_1():
	filename = input("Please enter the name of the data file:\n")
	data1 = pd.read_csv(filename, index_col=0)
	data2 = data1.select_dtypes(include = [np.number])		#string strip
	print("the following columns were imported:\n", list(data2))
	global pivot
	pivot = input("Enter the intended decision ID:\n")
	global classifier
	classifier = DecisionTreeClassifier(min_samples_split = 100)
	global features
	features = list(data2)
	global train, test
	train, test = train_test_split(data2, test_size = .1)
	features.remove(pivot)
	global x_train, y_train, X_test, y_test
	x_train = train[features]
	y_train = train[pivot]
	X_test = test[features]
	y_test = test[pivot]
	decisiontree = classifier.fit(x_train, y_train)
	filename = input("Please enter the name of the decision tree file:\n")
	jar = open(filename, "wb")
	jarf = open(filename+"features", "wb")
	jarp = open(filename+"pivot", "wb")
	jarc = open(filename+"class", "wb")
	filename = filename + '.png'
	savetree(decisiontree, features, filename)
	pickle.dump(decisiontree, jar)
	pickle.dump(features, jarf)
	pickle.dump(pivot, jarp)
	pickle.dump(classifier, jarc)
	jar.close()
	jarf.close()
	jarp.close()
	jarc.close()
	print("Tree saved!")
	main()

def case_2():
	y_pred = classifier.predict(X_test)
	score = accuracy_score(y_test, y_pred) * 100
	print("Accuracy: ", round(score, 1), "%")
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	main()

def case_3_1():
	print(list(features))
	df3in = input("Enter the corresponding data for the above fields, separated by commas:\n").split(',')
	df3in = [x.strip(' ') for x in df3in]
	df3list = []
	df3list.append(df3in)
	df3 = pd.DataFrame(df3list, columns=features)
	df3[pivot] = 0
	print(df3)
	y_pred = classifier.predict(df3[features])
	print(y_pred[0])
	if y_pred[0] == 0:
		print("Decision Tree determined True.\n\n")
	elif y_pred[0] == 1:
		print("Decision Tree determined False\n\n")
	case_3()

def case_3_2():
	main()

def case_3():
	print("1. Enter a new case")
	choice = input("2. Quit\n" )
	switch_case_3 = collections.defaultdict(lambda : case_6)
	switch_case_3['1'] = case_3_1
	switch_case_3['2'] = case_3_2
	switch_case_3[choice]()

def case_4():
	filename = input("Please enter the name of the decision tree file:\n")
	jarf = open(filename+"features", "rb")
	jarp = open(filename+"pivot", "rb")
	jarc = open(filename+"class", "rb")
	global features
	features = pickle.load(jarf)
	global pivot
	pivot = pickle.load(jarp)
	global classifier
	classifier = pickle.load(jarc)
	jarf.close()
	jarp.close()
	jarc.close()
	main()

def case_5():
	print("Exiting")
	exit()

def case_6():
	print("Please enter a valid menu option")
	main()

def main():
	print("Choose one of the following options:")
	print("1. Learn a decision tree and save the tree.")
	print("2. Test accuracy of the decision tree.")
	print("3. Applying the decision tree to new cases.")
	print("4. Load a tree model and apply to new cases interactively as in menu 3.")
	choice = input("5. Quit:\n")
	switch_main = collections.defaultdict(lambda : case_6)
	switch_main['1'] = case_1
	switch_main['2'] = case_2
	switch_main['3'] = case_3
	switch_main['4'] = case_4
	switch_main['5'] = case_5
	switch_main[choice]()

main()
