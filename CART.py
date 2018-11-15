"""



HEEEEEEEEEEEEEEEEEEEEEEEEEEEEY
DICKHEAAAAAAAAAAAAAAAAAAAAAAAAD
https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
REMOVE THIS LINK
YOU WASTE OF SPACE
OTHERWISE YOU'LL DEFINITELY GET CAUGHT
FOR PLAJURISM

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
allow for n fold cross validation

"""

# For reading CSV files & later code
import random
import csv


# For loading csv files
def csvLoader(f):
    f = open(f, "r")
    l = csv.reader(f)
    data = list(l)
    return data

# Randomly splits 'data' into 'nFolds' amount and creates lists of said splits
def crossValidationSplit(data, nFolds):
    dataSplit = list()
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

# Evaluates an algortihm using the crossValidationSplit function defined above
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


# Calculate the Gini Index for splitting the data set
def giniIndex(groups, classes):
    nInstances = sum([len(g) for g in groups])
    gini = 0
    for g in groups:
        size = len(g)
        if size == 0:
            continue
        score = 0
        for c in classes:
            p = [row[-1]for row in g].count(c)/size
            score += p*p
        gini += (1-score)* (size/nInstances)
    return gini

# Splits a dataset based on an attributes
def testingSplit(index, val, data):
    left, right = list(), list()
    for r in data:
        if r[index] < val:
            left.append(r)
        else:
            right.append(r)
    return left, right

# Select the best split for the data, by calculating the gini index for the data sets
def getSplit(data):
    c = list(set(row[-1] for row in data))
    splitIndex, splitValue, splitScore, splitGroups = 9999, 9999, 9999, None
    for i in range(len(data[0])-1):
        for row in data:
            groups = testingSplit(i, row[i], data)
            gini = giniIndex(groups, c)
            if gini < splitScore:
                splitIndex, splitValue, splitScore, splitGroups = i, row[i], gini, groups
    return {'index':splitIndex,'value':splitValue,'groups':splitGroups}

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

    # Left child
    if len(left) <= minSize:
        node['left'] = addToTerminal(left)
    else:
        node['left'] = getSplit(left)
        childNode(node['left'], maxDepth, minSize, depth+1)

    # Right child
    if len(right) <= minSize:
        node['right'] = addToTerminal(right)
    else:
        node['right'] = getSplit(right)
        childNode(node['right'], maxDepth, minSize, depth+1)

#Make decision tree
def makeDecisionTree(train, maxDepth, minSize):
    root = getSplit(train)
    childNode(root, maxDepth, minSize, 1)
    return root

# Make prediciton using decision tree
def prediction(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return prediction(node['left'], row)
        else:
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


# Testing on owls.csv

# Set Random seed
random.seed(7)

# Load data
file = (input("Please enter a file, or leave blank for owls.csv: ") or 'owls.csv')
data = csvLoader(file)

# evaluate algorithm
nFolds = 5
maxDepth = 5
minSize = 10
scores = algorithmEvaluation(data, cart, nFolds, maxDepth, minSize)

print('Scores: {}'.format(scores))
print('Mean Accuracy: %.2f%%' % (sum(scores)/(len(scores))))

input()
