"""
CT475 Assignment 3
Python implementation of CART algorithm.

Name: Taidgh Murray
Student ID: 15315901
Course: 4BS2


Create algorithm (Not Knn or NB)
Write own code
No use of libraries for ML implementation
Must use file input (import csv reader)


Distinguish between 'BarnOwl', 'SnowyOwl' & 'LongEaredOwl'
Divide into 2/3 training, 1/3 testing
Allow for n fold cross validation
Allow users rudimentary input if at all possible
Comment code for ease of reading, and to explain code decisions

The main elements of CART algorithm:

    Rules for splitting data at a node based on the value of one variable (Gini Impurity)

    Stopping rules for deciding when a branch is terminal and can be split no more (Values defined by user)

    A prediction for the target variable in each terminal node

"""

# For generating random seed (and other psuedorandom numbers)
import random
# For reading CSV files
import csv
# For printing run time at the end
import time


# For loading csv files
def csvLoader(f):
    f = open(f, "r")
    l = csv.reader(f)
    data = list(l)
    return data


# Randomly (psuedo-randomly) splits 'data' into 'nFolds' amount and creates lists of said splits
def crossValidationSplit(data, nFolds):
    dataSplit = list()
    # dataCopy variable implemented to avoid messy code
    dataCopy = list(data)
    foldSize = int(len(data)/nFolds)
    for j in range(nFolds):
        fold = list()
        while len(fold)<foldSize:
            index = random.randrange(len(dataCopy))
            fold.append(dataCopy.pop(index))
        dataSplit.append(fold)
    return dataSplit

# Accuracy percentage calculation - predicted value vs actual value
def accuracyCalculation(predicted, actual):
    correct = 0
    for k in range(len(actual)):
        if actual[k] == predicted [k]:
            correct+=1
    return ( correct / len(actual) ) * 100

# Evaluates algortihm using the crossValidationSplit function defined above
def algorithmEvaluation(data, algorithm, nFolds, *args):
    folds = crossValidationSplit(data, nFolds)
    score = list()
    for f in folds:
        trainingSet = list(folds)
        trainingSet.remove(f)
        trainingSet = sum(trainingSet, [])
        testingSet = list()
        for row in f:
            rowCopy = list(row)
            testingSet.append(rowCopy)
            rowCopy[-1] = None
        # Calculates predicition score for given algorithm
        pred = algorithm(trainingSet, testingSet, *args)
        act = [row[-1] for row in f]
        accuracy = accuracyCalculation(pred, act)
        score.append(accuracy)
    return score


# Calculate the quality of the data splits - Implementation of Gini Impurity
def splitQuality(groups, classes):
    nInstances = sum([len(g) for g in groups])
    splitQuality = 0
    for g in groups:
        size = len(g)
        # Avoids division by 0
        if size == 0:
            continue
        score = 0
        for c in classes:
            p = [row[-1]for row in g].count(c)/size
            score += p*p
        splitQuality += (1-score)* (size/nInstances)
    return splitQuality

# Splits a dataset based on an attributes
def testingSplit(index, val, data):
    left, right = list(), list()
    for r in data:
        if r[index] < val:
            left.append(r)
        else:
            right.append(r)
    return left, right

# Select the best split for the data, by calculating the splitQuality of the data sets
def getSplit(data):
    c = list(set(row[-1] for row in data))
    splitIndex, splitValue, splitScore, splitGroups = 999, 999, 999, None
    for i in range(len(data[0])-1):
        for row in data:
            # Calls the testingSplit function on the data
            groups = testingSplit(i, row[i], data)
            # Tests the split data for quality
            sQ = splitQuality(groups, c)
            # If the
            if sQ < splitScore:
                splitIndex, splitValue, splitScore, splitGroups = i, row[i], sQ, groups
    return {'index':splitIndex,'value':splitValue,'groups':splitGroups}

# Takes the group of rows assigned to a node and returns the most common value in the group, used to make predictions
def addToTerminal(group):
    outcome = [row[-1] for row in group]
    return max(set(outcome), key = outcome.count)

# Create child nodes for the decision tree
def childNode(node, maxDepth, minSize, depth):
    left, right = node['groups']
    del(node['groups'])

    # If no child nodes exist yet
    if not left or not right:
        node['left'] = node['right'] = addToTerminal(left + right)
        return

    # If the tree can't get any any larger, but the depht returns a larger value
    if depth >= maxDepth:
        node['left'], node['right'] = addToTerminal(left), addToTerminal(right)
        return

    # Left child node
    if len(left) <= minSize:
        # If the left node is smaller than the minimum size, its just added to the tree
        node['left'] = addToTerminal(left)
    else:
        # Otherwise it calls the getSplit function on itself
        node['left'] = getSplit(left)
        # And recursively calls the function on itself, increasing the depth by 1
        childNode(node['left'], maxDepth, minSize, depth+1)

    # Right child node
    if len(right) <= minSize:
        node['right'] = addToTerminal(right)
    else:
        node['right'] = getSplit(right)
        childNode(node['right'], maxDepth, minSize, depth+1)

# Generated initial decision tree
def makeDecisionTree(train, maxDepth, minSize):
    # Starts the tree with the best split of the training data
    root = getSplit(train)
    # Calls childNode function, which will recursively create binary tree from the root node
    childNode(root, maxDepth, minSize, 1)
    return root

# Make prediciton using decision tree
def prediction(node, row):
    # If the index node in the row is smaller than the value node
    if row[node['index']] < node['value']:
        # if the left node is a Python dictionary
        if isinstance(node['left'], dict):
            # Recursively calls the prediction function using the left node and the row
            return prediction(node['left'], row)
        else:
            # Otherwise just returns the left node
            return node['left']

    else:
        if isinstance(node['right'], dict):
            return prediction(node['right'], row)
        else:
            return node['right']


# Calling the CART algorithm
def cart(train, test, maxDepth, minSize):
    # Defines a tree using the makeDecisionTree funcion
    tree = makeDecisionTree(train, maxDepth, minSize)
    # Initialises empty list, called 'predictions' to hold predictions
    predictions = list()
    # Fills prediction list with predictions for each row of info in the test data
    for row in test:
        p = prediction(tree, row)
        predictions.append(p)
    return predictions



# Allows for some user input, they can chose a different file to be tested, the seed, and change the amount of folds, the maximum depth, and the minimum size of the tree



# Load data
file = (input("Please enter a file, or leave blank for owls.csv: ") or 'owls.csv')
data = csvLoader(file)

# Set Random seed
seed = (input("Please enter your desired seed, or leave blank for 7: ") or 7)
random.seed(seed)


# Evaluate algorithm inputs from user
nFolds = (input("Enter the number of folds you wish to create, or leave blank for default (3):") or 3)
maxDepth = (input("Enter the maximum depth of the tree, or leave blank for default (5):") or 5)
minSize = (input("Enter the minumum size of the tree, or leave blank for default (10):") or 10)

nFolds = int(nFolds)
maxDepth = int(maxDepth)
minSize = int(minSize)

# Record Start Time of prgram
startTime = time.time()

print('\n')
# CART algorithm
scores = algorithmEvaluation(data, cart, nFolds, maxDepth, minSize)

# Formats results and shows the classification accuracy for each fold
print('Accuracy scores for each fold: {}'.format(scores))
# Mean accuracy of accuracy score for CART Algorithm
print('Mean Accuracy: %.2f%%' % (sum(scores)/(len(scores))))

print("--- This calculation took %.2f seconds ---" % (time.time() - startTime))


# Users can then read the completed data before closing the program
input()
