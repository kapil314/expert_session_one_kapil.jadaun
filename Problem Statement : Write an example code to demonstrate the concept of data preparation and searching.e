## Importing important Libraries like numpy,pandas,seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## Fatching dataset from sklearn
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
## Loading dataset
wine=datasets.load_wine()
wine.keys()
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])
wine.feature_names
['alcohol',
 'malic_acid',
 'ash',
 'alcalinity_of_ash',
 'magnesium',
 'total_phenols',
 'flavanoids',
 'nonflavanoid_phenols',
 'proanthocyanins',
 'color_intensity',
 'hue',
 'od280/od315_of_diluted_wines',
 'proline']
wine.DESCR
'.. _wine_dataset:\n\nWine recognition dataset\n------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 178 (50 in each of three classes)\n    :Number of Attributes: 13 numeric, predictive attributes and the class\n    :Attribute Information:\n \t\t- Alcohol\n \t\t- Malic acid\n \t\t- Ash\n\t\t- Alcalinity of ash  \n \t\t- Magnesium\n\t\t- Total phenols\n \t\t- Flavanoids\n \t\t- Nonflavanoid phenols\n \t\t- Proanthocyanins\n\t\t- Color intensity\n \t\t- Hue\n \t\t- OD280/OD315 of diluted wines\n \t\t- Proline\n\n    - class:\n            - class_0\n            - class_1\n            - class_2\n\t\t\n    :Summary Statistics:\n    \n    ============================= ==== ===== ======= =====\n                                   Min   Max   Mean     SD\n    ============================= ==== ===== ======= =====\n    Alcohol:                      11.0  14.8    13.0   0.8\n    Malic Acid:                   0.74  5.80    2.34  1.12\n    Ash:                          1.36  3.23    2.36  0.27\n    Alcalinity of Ash:            10.6  30.0    19.5   3.3\n    Magnesium:                    70.0 162.0    99.7  14.3\n    Total Phenols:                0.98  3.88    2.29  0.63\n    Flavanoids:                   0.34  5.08    2.03  1.00\n    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12\n    Proanthocyanins:              0.41  3.58    1.59  0.57\n    Colour Intensity:              1.3  13.0     5.1   2.3\n    Hue:                          0.48  1.71    0.96  0.23\n    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71\n    Proline:                       278  1680     746   315\n    ============================= ==== ===== ======= =====\n\n    :Missing Attribute Values: None\n    :Class Distribution: class_0 (59), class_1 (71), class_2 (48)\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThis is a copy of UCI ML Wine recognition datasets.\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data\n\nThe data is the results of a chemical analysis of wines grown in the same\nregion in Italy by three different cultivators. There are thirteen different\nmeasurements taken for different constituents found in the three types of\nwine.\n\nOriginal Owners: \n\nForina, M. et al, PARVUS - \nAn Extendible Package for Data Exploration, Classification and Correlation. \nInstitute of Pharmaceutical and Food Analysis and Technologies,\nVia Brigata Salerno, 16147 Genoa, Italy.\n\nCitation:\n\nLichman, M. (2013). UCI Machine Learning Repository\n[https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,\nSchool of Information and Computer Science. \n\n.. topic:: References\n\n  (1) S. Aeberhard, D. Coomans and O. de Vel, \n  Comparison of Classifiers in High Dimensional Settings, \n  Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  \n  Mathematics and Statistics, James Cook University of North Queensland. \n  (Also submitted to Technometrics). \n\n  The data was used with many others for comparing various \n  classifiers. The classes are separable, though only RDA \n  has achieved 100% correct classification. \n  (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) \n  (All results using the leave-one-out technique) \n\n  (2) S. Aeberhard, D. Coomans and O. de Vel, \n  "THE CLASSIFICATION PERFORMANCE OF RDA" \n  Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of \n  Mathematics and Statistics, James Cook University of North Queensland. \n  (Also submitted to Journal of Chemometrics).\n'
wine.target_names
array(['class_0', 'class_1', 'class_2'], dtype='<U7')
x=wine.data
y=wine.target
## Making dataframe
df=pd.DataFrame(x,columns=wine.feature_names)
df['target']=y
df.head()
alcohol	malic_acid	ash	alcalinity_of_ash	magnesium	total_phenols	flavanoids	nonflavanoid_phenols	proanthocyanins	color_intensity	hue	od280/od315_of_diluted_wines	proline	target
0	14.23	1.71	2.43	15.6	127.0	2.80	3.06	0.28	2.29	5.64	1.04	3.92	1065.0	0
1	13.20	1.78	2.14	11.2	100.0	2.65	2.76	0.26	1.28	4.38	1.05	3.40	1050.0	0
2	13.16	2.36	2.67	18.6	101.0	2.80	3.24	0.30	2.81	5.68	1.03	3.17	1185.0	0
3	14.37	1.95	2.50	16.8	113.0	3.85	3.49	0.24	2.18	7.80	0.86	3.45	1480.0	0
4	13.24	2.59	2.87	21.0	118.0	2.80	2.69	0.39	1.82	4.32	1.04	2.93	735.0	0
df.tail()
alcohol	malic_acid	ash	alcalinity_of_ash	magnesium	total_phenols	flavanoids	nonflavanoid_phenols	proanthocyanins	color_intensity	hue	od280/od315_of_diluted_wines	proline	target
173	13.71	5.65	2.45	20.5	95.0	1.68	0.61	0.52	1.06	7.7	0.64	1.74	740.0	2
174	13.40	3.91	2.48	23.0	102.0	1.80	0.75	0.43	1.41	7.3	0.70	1.56	750.0	2
175	13.27	4.28	2.26	20.0	120.0	1.59	0.69	0.43	1.35	10.2	0.59	1.56	835.0	2
176	13.17	2.59	2.37	20.0	120.0	1.65	0.68	0.53	1.46	9.3	0.60	1.62	840.0	2
177	14.13	4.10	2.74	24.5	96.0	2.05	0.76	0.56	1.35	9.2	0.61	1.60	560.0	2
## Making pairplot to analyse
sns.pairplot(df)
<seaborn.axisgrid.PairGrid at 0x29b01950640>

## Decision Tree Classifier
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.25,stratify=y)
clasifier=DecisionTreeClassifier(criterion='gini',random_state=1)
clasifier.fit(x_train,y_train)
DecisionTreeClassifier(random_state=1)
y_predict=clasifier.predict(x_test)
## Checking Accurary
print("Accuracy",metrics.accuracy_score(y_test,y_predict))
Accuracy 0.8888888888888888
## Viewing Confusion Matrix
print("Confusion Matrix")
mat=metrics.confusion_matrix(y_test,y_predict)
mat
sns.heatmap(mat,annot=True,square=True,cbar=True,fmt='d',xticklabels=wine.target_names,yticklabels=wine.target_names)
plt.xlabel('Predicted')
plt.ylabel('True/Expected')
plt.show()
Confusion Matrix

Viewing Report of Data
print(metrics.classification_report(y_test,y_predict))
              precision    recall  f1-score   support

           0       0.88      0.93      0.90        15
           1       0.88      0.83      0.86        18
           2       0.92      0.92      0.92        12

    accuracy                           0.89        45
   macro avg       0.89      0.89      0.89        45
weighted avg       0.89      0.89      0.89        45

​
