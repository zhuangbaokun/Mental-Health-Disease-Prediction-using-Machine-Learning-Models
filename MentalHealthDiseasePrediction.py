# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import datetime as dt
import numpy as np
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
df = pd.read_csv("Mental_Health_Clean.csv")
df = df.drop(['Have you ever sought treatment for a mental health issue from a mental health professional?','Have you had a mental health disorder in the past?','Do you currently have a mental health disorder?','If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?','If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'],axis=1)
df.to_csv("Mental_Health_Clean_droppedObvColumns_NOTFEATURESELECTED.csv")

# Chi-squared  (categorical response, categorical predictors)
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfTabular = None
        self.dfExpected = None
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result=''
#             result="{0} is IMPORTANT for Prediction".format(colX)
            
        else:
#             result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
#             result="{0} is NOT an important predictor".format(colX)
            result=colX
            print(result)
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX, alpha)
cT = ChiSquare(df)
#Feature Selection

testColumns = df.columns.tolist()
for var in testColumns:
    cT.TestIndependence(colX=var,colY='Have you been diagnosed with a mental health condition by a medical professional?' )  
    
    
col_to_drop = ["How many employees does your company or organization have?"
,"Is your employer primarily a tech company/organization?"
,"Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?"
,"Does your employer offer resources to learn more about mental health concerns and options for seeking help?"
,"Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?"
,"Would you feel comfortable discussing a mental health disorder with your coworkers?"
,"Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"
,"Do you think that discussing a physical health issue with previous employers would have negative consequences?"
,"Would you have been willing to discuss a mental health issue with your previous co-workers?"
,"Would you be willing to bring up a physical health issue with a potential employer in an interview?"
,"Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"
,"What is your age?"]
# pd.DataFrame({"Unimportant Variables":col_to_drop}).to_csv("Dropped_Columns.csv")
df = df.drop(col_to_drop,axis=1)
df.to_csv("Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv")
df.shape
df = pd.read_csv("Mental_Health_Clean_droppedObvColumnsANDfeatureselected.csv").drop(["Unnamed: 0"],axis=1)
df.shape

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

#SearchCV
# from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import learning_curve,GridSearchCV,RandomizedSearchCV

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#Neural Network
from sklearn.neural_network import MLPClassifier


# Split Train Test
target=df['Have you been diagnosed with a mental health condition by a medical professional?']
y = target
X = df.drop(['Have you been diagnosed with a mental health condition by a medical professional?'],axis=1)
feature_cols = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.3, random_state=0)
# Import and Prepare Grid
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)



# Search for best Param
rf = RandomForestClassifier(n_estimators = 20)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
str(rf_random.best_params_).replace(":","=").replace("'","").replace("{","").replace("}","").replace("'","")

# Get Accuracy
rf = RandomForestClassifier(n_estimators= rf_random.best_params_['n_estimators'], min_samples_split= rf_random.best_params_['min_samples_split'], min_samples_leaf= rf_random.best_params_['min_samples_leaf'], max_features= rf_random.best_params_['max_features'], max_depth= rf_random.best_params_['max_depth'], bootstrap= rf_random.best_params_['bootstrap'])
rf.fit(X_train, y_train)
print(rf.score(X_test,y_test))

str(rf_random.best_params_).replace(":","=").replace("'","").replace("{","").replace("}","").replace("'","")



def evalClassModel(model, X, y, X_test, y_test, y_pred_class, plot=True):
    #Classification accuracy: percentage of correct predictions
    # calculate accuracy
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    # examine the class distribution of the testing set (using a Pandas Series method)
    print('Null accuracy:\n', y_test.value_counts())
    
    # calculate the percentage of ones
    print('Percentage of ones:', y_test.mean())
    
    # calculate the percentage of zeros
    print('Percentage of zeros:',1 - y_test.mean())
    
    #Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print('Classification Accuracy:', metrics.accuracy_score(y_test, y_pred_class))
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))
    
    #False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)
    
    #Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))
    
    
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))
    
    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
    
    ##########################################
    #Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    # 1D array (vector) of binary values (0, 1)
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(X_test)[0:10])

    # print the first 10 predicted probabilities for class 1
    model.predict_proba(X_test)[0:10, 1]
    
    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    if plot == True:
        # histogram of predicted probabilities
        # adjust the font size 
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)
        
        # x-axis limit from 0 to 1
        plt.xlim(0,1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
    y_pred_prob = y_pred_prob.reshape(-1,1) 
    y_pred_class = binarize(y_pred_prob, 0.3)[0]
    
    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()
        
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
    def evaluate_threshold(threshold):
        #Sensitivity: When the actual value is positive, how often is the prediction correct?
        #Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)
    return accuracy
evalClassModel(rf, X, y, X_test, y_test, rf.predict(X_test), plot=True)

# Tree classifier Important Variables
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
y = df['Have you ever sought treatment for a mental health issue from a mental health professional?']
X = df.drop(['Have you ever sought treatment for a mental health issue from a mental health professional?'],axis=1)
feature_cols=X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_
feature_importance = pd.DataFrame({"Features":X.columns.tolist(),"Gini Index":clf.feature_importances_})
feature_importance.sort_values("Gini Index",ascending=False,inplace=True)
# Top Ten Variable
sns.barplot(x="Gini Index",y="Features",data=feature_importance.iloc[1:11])



