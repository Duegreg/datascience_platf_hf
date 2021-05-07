# -*- coding: utf-8 -*-

# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
################################################################################
#                                                                              #
#____________________________PRE-PROCESSING_DATA_______________________________#
#                                                                              #
################################################################################
"""


# Importing the dataset
task = pd.read_csv('C:/Users/duegr/datasience_platf_hf/data/public_test.csv', index_col="ID")
data = pd.read_csv('C:/Users/duegr/datasience_platf_hf/data/public_train.csv', index_col="ID")

columns = set(data.columns)
print(columns)
target_col = "DEFAULT"
cat_cols = {'SECTOR', 'SUBSECTOR', 'CONTRACT_TYPE', 'CONTRACT_SUBTYPE', 'DIM_DENOM_DEV_ID', 
               'MODE', 'FREQ', 'PH', 'KKV', 'SIZE_CAT', 'ENTITY', 'MONTH'}
num_cols = columns.difference(cat_cols).difference({target_col})



"""
Handling missing values
"""
dtypes = [str(typ) for typ in data.dtypes.to_dict().values()]
info = list(zip(data.columns, data.isna().sum(), dtypes))

data.isna().sum()

def fillMissingValues(df):
    for num in num_cols:
        mean = df[num].mean()
        df[num] = df[num].fillna(mean)
        
    for cat in cat_cols:
        mode = df[cat].value_counts().idxmax()
        df[cat] = df[cat].fillna(mode) 
            
            
fillMissingValues(data) 
fillMissingValues(task)   
  
data.isna().sum()    



"""
Handling categorical data
"""

list(zip(data.columns, data.nunique().to_dict().values(), dtypes))
# data.info()


class CatHandler():
    
    def calculateWoeForData(self, df, cat_col):
        df_temp = (pd.crosstab(df[cat_col], df[target_col], normalize='columns'))
    
        df_temp["woe"] = df_temp.apply(lambda x: np.log(x[1] / x[0]), axis=1)
        df_temp["iv"] = df_temp.apply(lambda x: x["woe"] * np.sum(x[1] - x[0]), axis=1)
        df_temp.replace([np.inf, -np.inf], 0, inplace=True)
        return df_temp
            
    
    def calculateWoe(self, data, task, dummy_cols):
        for cat in cat_cols:
            if cat not in dummy_cols: 
                woe = self.calculateWoeForData(data, cat)     
                data[cat + "_IV"] = data[cat].apply(lambda x: woe["iv"].get(x, 0))
                task[cat + "_IV"] = task[cat].apply(lambda x: woe["iv"].get(x, 0))
            
            
    def handleCategoricalCols(self, data, task):
        dummy_cols = set([cat for cat in cat_cols if data[cat].nunique() <= 20])
        self.calculateWoe(data, task, dummy_cols)
        
        data = pd.get_dummies(data, columns=dummy_cols, drop_first=True)
        task = pd.get_dummies(task, columns=dummy_cols, drop_first=True)
        
        ordered_cols = sorted(list(set(task.columns).intersection(set(data.columns))))
        data = data[[target_col] + ordered_cols] # drop and order columns
        task = task[ordered_cols] # drop and order columns
        
        return data, task
        
        
            
    
catHandler = CatHandler()    
data, task = catHandler.handleCategoricalCols(data, task)
train_cols = data.columns.difference(cat_cols).difference({target_col})

data.info()


# np.random.seed(100)
# df = pd.DataFrame({'grade': np.random.choice(list('ABCD'),size=(20)),
#                     'pass': np.random.choice([0,1],size=(20))
# })
# feature, target = 'grade','pass'
# df_temp = (pd.crosstab(df[feature], df[target], normalize='columns'))
# df_temp["woe"] = df_temp.apply(lambda x: np.log(x[1] / x[0]), axis=1)
# df_temp["iv"] = df_temp.apply(lambda x: x["woe"] * np.sum(x[1] - x[0]), axis=1)
# print(df_temp)
# df_temp["woe"]["D"]

"""
Feature Scaling
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(data[train_cols])
X_task = sc.transform(task[train_cols])
y_target = data[target_col].values





"""
Train and predict
"""


n_features = {4, 8, 16, 32, -1}
feature_mode = {"raw", "pca", "kpca", "lda"} #, "pca-lda", "lda-pca", "kpca-lda", "lda-kpca"}

# Applying PCA
# nem csökkneti a featureket, hanem újat hoz létre beőlük (most épp 2-t, ezek a principal compok)
from sklearn.decomposition import PCA  
pca = PCA(n_components=2,
          random_state=42)
X_train = pca.fit_transform(X_train)
X_task = pca.transform(X_task)

kernel = {'linear', 'poly', 'rbf', 'sigmoid'}
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, 
                 kernel='rbf',
                  shrinkage='auto',
                 n_jobs=-1)
X_train = kpca.fit_transform(X_train)
X_task = kpca.transform(X_task)


solver = {'svd', 'lsqr', 'eigen'}
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2,
          solver='svd',
          shrinkage='auto')
X_train = lda.fit_transform(X_train, y_target)
X_task = lda.transform(X_task)




neighbors = {3, 5, 10, 20}
# Training the K-NN model on the Training set
#minkowski with p=2 is equivalent to the standard Euclidean metric (ezek a defaultak)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, 
                                  metric="minkowski", 
                                  p=2, 
                                  n_jobs=-1)

LogisticRegression_params = {'C': [0.1, 0.5, 1, 2, 5]}
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42, 
                                max_iter=1000, 
                                class_weight='balanced', 
                                n_jobs=-1)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', 
                                    random_state=42, 
                                    max_features='auto', 
     
                                    
     class_weight='balanced')
RandomForestClassifier_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 25],
    'max_leaf_nodes': [None, 100, 200]
}
n_estimators = {10, 50, 100}
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, 
                                    criterion='entropy',
                                    max_features="auto",
                                    class_weight='balanced',
                                    n_jobs=-1,
                                    random_state=42)


SVC_params = [
    {'C': [0.1, 0.5, 1, 2, 5], 'kernel': ['linear']}, #két dictionary a gamma miatt
    {'C': [0.1, 0.5, 1, 2, 5], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [0.1, 0.5, 1, 2, 5]}
]
# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', 
                 random_state=42,
                 class_weight='balanced',
                 probability=True)


XGBC_params = {
        'min_child_weight': [1, 2, 50],
        'gamma': [0.1, 0.5, 1, 2, 5],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 10],
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.05, 0.1, 0.2, 0.5],
        
        'reg_alpha': [0, 1, 1.2],
        'reg_lambda': [1, 1.2],
        'subsample': [0.7, 0.8, 0.9, 1],
        'seed': 42,
        'random_state': 42,
        'n_jobs': -1
        }
# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(n_jobs=-1, 
                           random_state=42)



X_train.shape
y_target.shape

classifier.fit(X_train, y_target)
pred = classifier.predict_proba(X_task)




check = pd.DataFrame(list(zip(y_target, pred)), columns=["target", "prediction"])
from sklearn.metrics import confusion_matrix, accuracy_score
self_pred = classifier.predict(X_train)
# TN FP
# FN TP
print(confusion_matrix(y_target, self_pred))
accuracy_score(y_target, self_pred)

from sklearn.metrics import roc_auc_score, classification_report
roc_auc_score(y_target, self_pred)
# fpr, tpr, thresholds = roc_curve(y_target, self_pred)
# auc(fpr, tpr)
print(classification_report(y_target, self_pred))

# pred = classifier.decision_function(X_task) # ezt még normalizálni kéne?



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_target, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))



# Applying Grid Search to find the best model and the best parameters
#C for reducing overfitting (lower C stronger regularization), 
#gamma: nem tudom
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']}, #két dictionary a gamma miatt
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy', #mivel classification scoring='roc_auc',
                           cv = 10, #built in cross-validation
                           n_jobs = -1) #all the processors will be use

grid_search.fit(X_train, y_target) #csak trainig!!!

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_




"""
Saving results
"""

results = pd.DataFrame(list(zip(task.index.values, pred[:,-1])), columns=["ID", "PROBA_OF_DEFAULT"])
results.to_csv("C:/Users/duegr/datasience_platf_hf/result.csv", index=False)
results.head()






"""
1. missing values
1.1. kategórikus változók
4. standardization

3. feature selection (PCA, LDA...?)


11. grid search (cross val) minden típusra
"""




# X = train_data.iloc[:, 1:].values
# y = train_data.iloc[:, 0].values



# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Training the Decision Tree Classification model on the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)

