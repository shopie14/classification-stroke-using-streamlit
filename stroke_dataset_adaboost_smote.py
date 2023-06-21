# -*- coding: utf-8 -*-
"""Stroke Dataset Adaboost SMOTE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iBffdEdRU57NloPGu-CvlLrmid7uG-eE

Kelompok 1: 
1. Shopi Nurhidayanti 
2. Anisa Syifa Syafaat
3. Dini Antika 
4. Dewi Sulastri
5. Muhammad Ilyasa

Dataset Stroke adalah kumpulan data yang berisi informasi tentang pasien-pasien yang didiagnosis dengan stroke. Setiap sampel dalam dataset ini memiliki beberapa fitur atau atribut yang menggambarkan karakteristik pasien, seperti usia, jenis kelamin, tekanan darah, kadar gula darah, indeks massa tubuh, dan sebagainya. Selain itu, setiap sampel juga memiliki label yang menunjukkan apakah pasien mengalami stroke atau tidak.
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive/MATA KULIAH 😊😊/SEMESTER 6/PRAKTIKUM KECERDASAN BUATAN/TUGAS BESAR/Dataset"

"""## Import Modul"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score 
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

"""## Data Understanding"""

data = pd.read_csv('stroke.csv')
data

data.info()

data.isnull().sum()

data.shape

data.describe()

data = data.drop_duplicates()
data.shape

data.info()

data.dropna(inplace = True)

data['stroke'].value_counts()

data.info()

"""##  Exploratory Data Analysis"""

countplot_cols = ['heart_disease', 'hypertension', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

boxplot_cols = ['age','avg_glucose_level', 'bmi']

for i, column in enumerate(countplot_cols):
    sns.countplot(x=column, hue = 'stroke', data=data)
    plt.show()

for i, column in enumerate(boxplot_cols):
    sns.boxplot(x='stroke', y=column, data=data)
    plt.show()

data = data.drop(data[data.gender == 'Other'].index)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])
data = data.drop('id', axis = 1)
print('Encoding was successful ')

import numpy as np
def feature_creation(data):
    data['age1'] = np.log(data['age'])
    data['age2'] = np.sqrt(data['age'])
    data['age3'] = data['age']**3
    data['bmi1'] = np.log(data['bmi'])
    data['bmi2'] = np.sqrt(data['bmi'])
    data['bmi3'] = data['bmi']**3
    data['avg_glucose_level1'] = np.log(data['avg_glucose_level'])
    data['avg_glucose_level2'] = np.sqrt(data['avg_glucose_level'])
    data['avg_glucose_level3'] = data['avg_glucose_level']**3
    for i in ['gender', 'age1', 'age2', 'age3', 'hypertension', 'heart_disease', 'ever_married', 'work_type']:
        for j in ['Residence_type', 'avg_glucose_level1','avg_glucose_level2', 'avg_glucose_level3', 'bmi1', 'bmi2', 'bmi3','smoking_status']:
            data[i+'_'+j] = data[i].astype('str')+'_'+data[j].astype('str')
    return data

data = feature_creation(data)

data.shape

# Determination categorical features

categorical_columns = []
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
features = data.columns.values.tolist()
for col in features:
    if data[col].dtype in numerics: continue
    categorical_columns.append(col)

categorical_columns

# Encoding categorical features

for col in categorical_columns:
    if col in data.columns:
        #le = LabelEncoder()
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))
print('Encoding was successfull')

#num_features_opt = 25 
num_features_opt = 40 


# the somewhat excessive number of features, which we will choose at each stage
num_features_max = 50   

features_best = []
X_train = data.drop('stroke', axis = 1).copy()
y_train = data.stroke.copy()

"""## Feature Selection"""

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

"""### Feature Selection with Pearson correlation"""

# Threshold for removing correlated variables
threshold = 0.9

def highlight(value):
    if value > threshold:
        style = 'background-color: pink'
    else:
        style = 'background-color: green'
    return style

# Absolute value correlation matrix
corr_matrix = data.corr().abs().round(2)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.style.format("{:.2f}").applymap(highlight)

# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop features with correlation above the threshold
features_filtered = data.drop(columns = collinear_features)
print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])

# Add filtered features
features_best.append(features_filtered.columns.tolist())

"""### Feature selection by the SelectFromModel with LinearSVC"""

lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_train)
X_selected_df = pd.DataFrame(X_new, columns=[X_train.columns[i] for i in range(len(X_train.columns)) if model.get_support()[i]])

"""### Feature selection with Lasso"""

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=3).fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_new = model.transform(X_train)
X_selected_df = pd.DataFrame(X_new, columns=[X_train.columns[i] for i in range(len(X_train.columns)) if model.get_support()[i]])

# add features
features_best.append(X_selected_df.columns.tolist())

"""### Feature selection by the SelectKBest with Chi-2"""

from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(abs(X_train), y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']

features_best.append(featureScores.nlargest(num_features_max,'Score')['Feature'].tolist())
print(featureScores.nlargest(len(dfcolumns),'Score'))

features_best

main_cols_max = features_best[0]
for i in range(len(features_best)-1):
    main_cols_max = list(set(main_cols_max) | set(features_best[i+1]))
print(main_cols_max)

print('Cols:', len(main_cols_max))

main_cols = []
main_cols_opt = {feature_name : 0 for feature_name in data.columns.tolist()}
for i in range(len(features_best)):
    for feature_name in features_best[i]:
        main_cols_opt[feature_name] += 1
data_main_cols_opt = pd.DataFrame.from_dict(main_cols_opt, orient='index', columns=['Num'])
data_main_cols_opt.sort_values(by=['Num'], ascending=False).head(num_features_opt)

# Select only our best features that are included in num_features_opt
main_cols = data_main_cols_opt.nlargest(num_features_opt, 'Num').index.tolist()
if not 'stroke' in main_cols:
    main_cols.append('stroke')
print(main_cols)

print("Quantity:", len(main_cols))

data[main_cols].head()

"""## Balance with SMOTE"""

X = data[main_cols].drop('stroke', axis = 1)
y = data[main_cols].stroke

#scaling
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()

X_rs = pd.DataFrame(rs.fit_transform(X), columns = X.columns)

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_rs, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=42)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
print(y_train_sm.value_counts())

"""## Modeling

## Decision Tree
"""

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
# Buat Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Latih Decision Tree Classifier
decision_tree.fit(X_train_sm, y_train_sm)

# Predict the labels for the test set using the combined model
y_pred_dt = decision_tree.predict(X_test)

# Membuat laporan klasifikasi
classification_dt = classification_report(y_test,y_pred_dt)

# Calculate the accuracy of the combined model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy:", accuracy_dt)

# Menampilkan laporan klasifikasi
print(classification_dt)

"""## Adaboost"""

from sklearn.ensemble import AdaBoostClassifier

# Membuat model Adaboost
adaboost = AdaBoostClassifier(random_state=42)

# Melatih model Adaboost dengan data yang telah di-sampling menggunakan metode "ACTUAL SMOTE"
adaboost.fit(X_train_sm, y_train_sm)

# Predict the labels for the test set using the combined model
y_pred2 = adaboost.predict(X_test)

# Membuat laporan klasifikasi
classification_rep2 = classification_report(y_test, y_pred2)

# Calculate the accuracy of the combined model
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy2)

# Menampilkan laporan klasifikasi
print(classification_rep2)

"""## C45 & Adaboost"""

from sklearn.tree import DecisionTreeClassifier
# Create a C4.5 decision tree classifier
c45 = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Train the C4.5 decision tree
c45.fit(X_train_sm, y_train_sm)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

# Create an AdaBoost classifier with the C4.5 decision tree as the base estimator
adaboost = AdaBoostClassifier(base_estimator=c45, random_state=42)

# Perform cross-validation with 5 folds
scores = cross_val_score(adaboost, X_train_sm, y_train_sm, cv=5)

# Print the accuracy scores from cross-validation
print("Cross-Validation Accuracy Scores:", scores)

# Print the mean accuracy and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Obtain cross-validated predictions
y_pred_cv = cross_val_predict(adaboost, X_train_sm, y_train_sm, cv=5)

# Generate classification report
report = classification_report(y_train_sm, y_pred_cv)

# Print the classification report
print(report)