"""
Technical university of Denmark
02450 - Introduction to Machine Learning and Data Mining
Group project 2
20-04-2021

This script is for the classification part of the project 2 report.
Parts from multiple exercise scripts from the course have been used for this script.

Author: Emil Priess Nielsen s193881

"""
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from platform import system
import numpy as np
import pandas as pd
import os
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread
from toolbox_02450 import *

#Import data
cwd = os.getcwd()
temp_dir = os.path.join(cwd, 'spambase.data')
data = pd.read_csv(temp_dir)
spam = np.copy(data)
N, M = spam.shape

#Standardization of dataset
X = spam[:,0:57]
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma
y = spam[:,-1]

#################################
#Test run of logistic regression#
#################################

#Test run on all of the data
X_train = X
X_test = X
y_train = y
y_test = y

#Creates regularization strength interval
lambda_interval = np.logspace(-8, 2, 50)
#Initialises arrays
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

#Runs logistic regression model for each value in lambda interval
for k in range(0, len(lambda_interval)):
    mdl = lm.LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

#Computes error and optimal lambda
min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]
plt.figure(figsize=(8,8))

#Plots error rate as a function of regularization strength
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 8, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 20])
plt.grid()
plt.show()    

#Plots parameter L2 norm as a function of regularization strength
plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

#################################
#Test run of classification tree#
#################################

#Creates attribute names
attributeNames = ['make','address','all','3d','our','over','remove','internet','order','mail','recieve','will','people','report','addresses'
              ,'free','business','email','you','credit','your','font','000','money','hp','hpl','george','650','lab','labs','telnet','857'
              ,'data','415','85','technology','1999','parts','pm','direct','cs','meeting','original','project','re','edu','table','conference'
              ,';','(','[','!','$','#','CRL_Avg','CRL_Longest','CRL_Total']     

#Declares the decision tree criterion
criterion='gini'
#Declares and fits the classification tree
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=(100))
dtc = dtc.fit(X,y)
#fname='tree_' + criterion + '_spam_emails100'
# Export tree graph .gvz file to parse to graphviz
#out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)

#Predicts y-values from the model
y_train_est_dtc = dtc.predict(X_train).T
y_test_est_dtc = dtc.predict(X_test).T




############################
#Two-level cross-validation#
############################

k = 0
CV = model_selection.KFold(10, shuffle=True)

gen_error_log = np.zeros(10)
gen_error_base = np.zeros(10)
gen_error_tree = np.zeros(10)
yhat = []
y_true = []
r1 = []
r2 = []
r3 = []

#Creates regularization strength interval
lambda_log_interval = np.logspace(-5, 1, 10)

#Outer loop
for par_index, test_index in CV.split(X, y):
    #Extract paring and test set for current CV fold
    X_par = X[par_index]
    y_par = y[par_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    y_val_error_log = np.zeros(10)
    y_val_error_base = np.zeros(10)
    y_val_error_tree = np.zeros(10)
    
    i=0
    CV_2 = model_selection.KFold(10, shuffle=True)
    
    #Inner loop
    for train_index, val_index in CV_2.split(X_par):
        #Extract train and validation set for current inner fold
        X_train = X_par[train_index]
        y_train = y_par[train_index]
        X_val = X_par[val_index]
        y_val = y_par[val_index]
        
        ###### Logistic ######
        logistic = lm.LogisticRegression(penalty='l2', C=1/lambda_log_interval[k])
        logistic.fit(X_train,y_train)
        y_val_est_log = logistic.predict(X_val)
        y_val_error_log[i] = np.sum(y_val_est_log != y_val) / len(y_val)
        
        
        ###### Baseline ######
        spamOccur = np.count_nonzero(y_train == 1)
        notSpamOccur = np.count_nonzero(y_train == 0)
        
        if (spamOccur >= notSpamOccur):
            c = 1
            
        elif (spamOccur < notSpamOccur):
            c = 0
        
        baselineY = np.full(len(y_val), c)
        
        errorCount = 0
        for l in range(len(y_val)):
            if (baselineY[l] != y_val[l]):
                errorCount = errorCount + 1
        
        y_val_error_base[i] = errorCount/len(y_val)

        
        ###### ClassTree #####
        classTree = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=(10 + k*110))
        classTree.fit(X_train,y_train)
        
        y_val_est_tree = classTree.predict(X_val)
        y_val_error_tree[i] = np.sum(y_val_est_tree != y_val) / len(y_val)
        
        
        #Increment inner loop
        i = i + 1
    
    #Model generalization errors    
    gen_error_log[k] = np.mean(y_val_error_log)
    gen_error_base[k] = np.mean(y_val_error_base)
    gen_error_tree[k] = np.mean(y_val_error_tree) 
    
    best_log = np.argmin(y_val_error_log)
    best_tree = np.argmin(y_val_error_tree)
    
    #############################
    #Statistical evaluation runs#
    #############################
    
    #Baseline
    spamOccur = np.count_nonzero(y_par == 1)
    notSpamOccur = np.count_nonzero(y_par == 0)
    
    if (spamOccur >= notSpamOccur):
        c = 1
        
    elif (spamOccur < notSpamOccur):
        c = 0
    
    yhatA = np.full(len(y_test), c)
    
    #Logistic regression
    logStat = lm.LogisticRegression(penalty='l2', C=1/lambda_log_interval[best_log])
    logStat.fit(X_par,y_par)
    yhatB = logistic.predict(X_test)

    
    #Classification tree
    statTree = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=(10 + best_tree*110))
    classTree.fit(X_par,y_par)
    yhatC = classTree.predict(X_test)
    
    y_true.append(y_test)
    
    r1.append( np.mean( np.abs( yhatA-y_test ) ** 2 - np.abs( yhatB-y_test) ** 2 ) )
    r2.append( np.mean( np.abs( yhatA-y_test ) ** 2 - np.abs( yhatC-y_test) ** 2 ) )
    r3.append( np.mean( np.abs( yhatC-y_test ) ** 2 - np.abs( yhatB-y_test) ** 2 ) )
    
    #Increment outer loop
    k = k + 1 
    
#Values for the complexity-controlling parameter
complexParam = np.array([10, 120, 230, 340, 450, 560, 670, 780, 890, 1000])
    
#Creates table for two-level cross-validation for the report
table = np.c_[np.arange(1,11), complexParam, gen_error_tree, lambda_log_interval, gen_error_log, gen_error_base]
table = pd.DataFrame(table, columns = ['Outer fold', 'Complexity parameter', 'Error_tree', 'Regularization strength', 'Error_logistic', 'Error_baseline'])



########################
#Statistical evaluation#
########################

#Initialize parameters and run test appropriate for statistical evaluation setup II
alpha = 0.05
rho = 1/10
p_setupIIAB, CI_setupIIAB = correlated_ttest(r1, rho, alpha=alpha)
p_setupIIAC, CI_setupIIAC = correlated_ttest(r2, rho, alpha=alpha)
p_setupIICB, CI_setupIICB = correlated_ttest(r3, rho, alpha=alpha)



########################################
#Training new logistic regression model#
########################################

#Randomly creates test and training set by a 0.5 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

#Creates logistic regression model with optimal regularization strength from cross-validation
newModel = lm.LogisticRegression(penalty='l2', C=1/0.0215)
newModel.fit(X_train, y_train)

#Classifies email as non-spam/spam (0/1) and assess probabilities
y_est = newModel.predict(X_test)
y_est_nonSpam_prob = newModel.predict_proba(X_test)[:, 0] 

#Evaluates classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y_test) / float(len(y_est))

#Displays classification results
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

#Plots the predictions for the observations in the test set
f = plt.figure();
class0_ids = np.nonzero(y_test==0)[0].tolist()
plt.plot(class0_ids, y_est_nonSpam_prob[class0_ids], '.y')
class1_ids = np.nonzero(y_test==1)[0].tolist()
plt.plot(class1_ids, y_est_nonSpam_prob[class1_ids], '.r')
plt.xlabel('Data object (email)'); plt.ylabel('Predicted prob. of class Not Spam');
plt.legend(['Not Spam', 'Spam'])
plt.ylim(-0.01,1.5)

plt.show()