from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Script to predict results for a dataset using sklearn
# built-in classifiers. 

# Decision tree flow chart to store data.
# Receive Yes or No for each node, to direct the data.

# [height, weight, shoe_size] 
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#List of labels associated with prvious list X
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Test data
testData = [[190, 80, 42], [155,50,33], [160, 78, 39]]
testTrue = ['male', 'female', 'male']

#Store Decision tree classifier for three classifiers 
#1. Tree
#2. Gaussian
#3 Quadratic Discriminant 
clfTree = DecisionTreeClassifier()
clfGaus = GaussianProcessClassifier()
clfQuad = QuadraticDiscriminantAnalysis()


#Trains the decision tree on the Dataset
clfTree = clfTree.fit(X,Y)
clfGaus = clfGaus.fit(X,Y)
clfQuad = clfQuad.fit(X,Y)

#Prediction of gender 
predTree = clfTree.predict(testData)
predGaus = clfGaus.predict(testData)
predQuad = clfQuad.predict(testData)

#Check the accuracy score 
treeScore = accuracy_score(testTrue, predTree)
gausScore = accuracy_score(testTrue, predGaus)
quadScore = accuracy_score(testTrue, predQuad)

#Print the prediction 
print("Decision Tree Classifier Prediction:")
print(predTree)
print("Gaussian Process Classifier Prediction:")
print(predGaus)
print("Quadratic Discriminant Analysis Prediction:")
print(predQuad)

#Print the accuracy score 
print("Decision Tree Classifier Accuracy:")
print(treeScore)
print("Gaussian Process Classifier Accuracy:")
print(gausScore)
print("Quadratic Discriminant Analysis Accuracy:")
print(quadScore)