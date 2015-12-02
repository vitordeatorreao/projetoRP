import sys, os
import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def parse_poli(df):
	"""
	Creates new features by applying polinomials to the existing ones.

	Parameters
	==========
	df : pandas.DataFrame
		A DataFrame containing the data set
	"""
	df["poli1"] = df["dia_semana_id"]**2 * df["destrito_id"]**3
	normalize(df["poli1"], copy=False)
	df["poli2"] = df["hrs"]**3 * df["destrito_id"]**2
	normalize(df["poli2"], copy=False)
	df["poli3"] = df["rua"]**2 * df["dia_semana_id"]**3
	normalize(df["poli3"], copy=False)
	df["poli4"] = df["latitude"]**3 * df["dia_semana_id"]**2
	normalize(df["poli4"], copy=False)
	df["poli5"] = df["longitude"]**3 * df["dia_semana_id"]**2
	normalize(df["poli5"], copy=False)

# load training Data Frame
trainDF = pd.read_csv(os.getcwd() + os.sep + ".." + \
    os.sep + "datasets" + os.sep + "experimentFeaturizedAddress.csv")
# get the class labels
labels = trainDF["categoria"].astype('category')
#get the list of addresses ordered:
addresses=sorted(trainDF["endereco"].unique())
#get the list of categories ordered:
categories=sorted(trainDF["categoria"].unique())
#count the number of categories
C_counts=trainDF.groupby(["categoria"]).size()
#count the number of unique lines where address and category are the same
A_C_counts=trainDF.groupby(["endereco","categoria"]).size()
#count the number of unique addresses
A_counts=trainDF.groupby(["endereco"]).size()
#initialize log_odds calculus
logodds={}
logoddsPA={}
MIN_CAT_COUNTS=2
#base logodds
default_logodds=np.log(C_counts/len(trainDF))-\
                np.log(1.0-C_counts/float(len(trainDF)))
# calculate log_odds for the address
for addr in addresses:
    PA=A_counts[addr]/float(len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and \
                        A_C_counts[addr][cat]<A_counts[addr]:
            PA=A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
    logodds[addr]=pd.Series(logodds[addr])
    logodds[addr].index=range(len(categories))
# get the polinomials
#parse_poli(trainDF)
trainDF["logoddsPA"] = trainDF["endereco"].apply(lambda x: logoddsPA[x])
address_features = trainDF["endereco"].apply(lambda x: logodds[x])
address_features.columns = ["logodds"+str(x) \
                for x in range(len(address_features.columns))]
feature_list = trainDF.columns.tolist()
feature_list.remove("categoria")
feature_list.remove("endereco")
feature_list.remove("dia_semana_id")
feature_list.remove("destrito_id")
data_set = trainDF[feature_list].join(address_features.ix[:,:])

splitTrain = StratifiedShuffleSplit(labels, train_size=0.7)
for train_index, test_index in splitTrain:
    train_set,test_set=data_set.iloc[train_index],data_set.iloc[test_index]
    labels_train,labels_test=labels[train_index],labels[test_index]
test_set.index=range(len(test_set))
train_set.index=range(len(train_set))
labels_train.index=range(len(labels_train))
labels_test.index=range(len(labels_test))
data_set.index=range(len(data_set))
labels.index=range(len(labels))

# model = LogisticRegression()
# model.fit(train_set,labels_train)

# print("all", \
#     log_loss(labels, model.predict_proba(data_set.as_matrix())))
# print("train", \
#     log_loss(labels_train, model.predict_proba(train_set.as_matrix())))
# print("test logloss", \
#     log_loss(labels_test, model.predict_proba(test_set.as_matrix())))

# print("logistic","accuracy = \n", \
#     cross_validation.cross_val_score(model, data_set, labels, cv=5, scoring='accuracy'),"\n")

# print("logistic","log_loss = \n", \
#     cross_validation.cross_val_score(model, data_set, labels, cv=5, scoring='log_loss'),"\n")

# model2 = GaussianNB()

# print("naive bayes","accuracy = \n", \
#     cross_validation.cross_val_score(model2, data_set, labels, cv=5, scoring='accuracy'),"\n")

# print("naive bayes","log_loss = \n", \
#     cross_validation.cross_val_score(model2, data_set, labels, cv=5, scoring='log_loss'),"\n")

model3 = SVC(probability=True)

print("naive bayes","accuracy = \n", \
    cross_validation.cross_val_score(model3, data_set, labels, cv=5, scoring='accuracy'),"\n")

print("naive bayes","log_loss = \n", \
    cross_validation.cross_val_score(model3, data_set, labels, cv=5, scoring='log_loss'),"\n")