# Import all the necessary libraries to start 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import binarize 

# Load the dataset and take a look at the first five rows
bank = pd.read_csv("bankfull.csv")
bank.head()

# Use min/max feature scaling to normalize the dataset 
minmaxscaler = preprocessing.MinMaxScaler(feature_range = (0,1))
bank = minmaxscaler.fit_transform(bank)
bank = pd.DataFrame(bank, columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y'])

# Take a look and inspect the results of the min/max feature scaling to ensure everything is in order 
bank.head()

# Separate the data into x and y 
bank_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome']
x = bank[bank_cols]
y = bank.y

# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0) 

# Fit the training dataset to logistic regression model 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Take a look at the model's accuracy rate 
y_pred_class = lr.predict(x_test)
from sklearn import metrics 
print ("The Accuracy Rate is", (metrics.accuracy_score(y_test, y_pred_class)*100))

# Import libraries necessary for building the confusion matrix 
# I left off "inline" so Jupyter could display the charts and graphs in separate windows 
import itertools
from sklearn.metrics import confusion_matrix
%matplotlib

# Create the confusion matrix and the graph used to display all of the information 
cm = confusion_matrix(y_test, y_pred_class)
plt.clf()
plt.figure(figsize = (9, 7))
plt.imshow(cm, interpolation='nearest', cmap = 'Blues')
plt.colorbar()
classNames = ['No','Yes']
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=0)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
thresh = cm.max() / 1.5 if s else cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+format(cm[i][j]), 
                 horizontalalignment ="center", 
                 color ="white" if cm[i][j] > thresh else "black")       
plt.tight_layout()
plt.show()

# Confusion Matrix Stats
TN = 1069 
FN = 172 
FP = 79
TP = 203

# Classification Table Metrics
Sensitivity = (TP/(TP+FN)) 
Specificity = (TN/(TN+FP))
Precision = (TP/(TP+FP))
False_Positive_Rate = (FP/251)
Misclassification_Rate = (FP+FN)/(1523)
Accuracy = (TP + TN)/(1523)

# Put the confusion matrix stats and metrics into a dataframe for easy reading and interpretation
cm_d = {'Sensitivity': [(TP/(TP+FN))*100], 
'Specificity': [(TN/(TN+FP))*100], 
'Precision': [(TP/(TP+FP))*100], 
'False_Positive_Rate': [(FP/251)*100], 
'Misclassification_Rate': [(FP+FN)/(1523)*100], 
'Accuracy': [(TP + TN)/(1523)*100]}

cm_metrics = pd.DataFrame(data = cm_d)
cm_metrics.transpose()

# Save all the probabilistic outcomes and create a histogram of the predictions
plt.figure(figsize = (9, 7))
bond_prob = lr.predict_proba(x_test)[:, 1]
plt.hist(bond_prob, edgecolor = 'gray', bins = 10)
plt.xlim(0, 1)
plt.ylabel('Frequency')
plt.xlabel('Predicted Propensities for Term Deposit')
plt.title('Histogram')

# Import the libraries necessary to plot the ROC Curve 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
%matplotlib

fpr, tpr, thresholds = metrics.roc_curve(y_test, bond_prob)
roc_auc = auc(fpr, tpr)

plt.clf()
plt.figure(figsize = (9, 7))
plt.plot(fpr, tpr, color='navy',
         lw=2, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 11
plt.title('ROC Curve')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print the AUC or area under the curve  
print(metrics.roc_auc_score(y_test, bond_prob))

# Test out different thresholds for the model
def th(threshold):
    print ('Sensitivity:', tpr[thresholds > threshold][-1])
    print ('Specificity:', 1 - fpr[thresholds > threshold][-1])

th(0.5)
th(0.4)
th(0.3)
