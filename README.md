# Gender_Classification

The code written in Python 3 uses the [scikit-learn](http://scikit-learn.org/) machine learning library to train a [decision tree](https://en.wikipedia.org/wiki/Decision_tree) on a small dataset of body metrics (height, width, and shoe size) labeled male or female. Then we can predict the gender of someone given a novel set of body metrics. 

Three classifiers were used to make predictions:
- [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier "View documentation for sklearn.tree.DecisionTreeClassifier")
-  [QuadraticDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis "View documentation for sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis")
- [GaussianProcessClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier "View documentation for sklearn.gaussian_process.GaussianProcessClassifier")

The accuracy score for each classifier was calculated using Sklearn scoring module [accuracy_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) which is printed out to the terminal for the user to review along with prediction from each classifier. 

## Dependencies
-   [Scikit-learn](http://scikit-learn.org/stable/install.html)
-   [numpy](http://www.numpy.org/)

To install these dependencies with [**pip**](https://pypi.org/project/pip/) simply run the following commands:

```bash 
sudo pip install -U pip
sudo pip install -U scikit-learn
```
