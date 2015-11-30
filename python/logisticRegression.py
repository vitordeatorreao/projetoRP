import sys, os
import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# load training Data Frame
trainDF = pd.read_csv(os.getcwd() + os.sep + ".." + \
    os.sep + "datasets" + os.sep + "trainFeaturizedAddress.csv")
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

trainDF["logoddsPA"] = trainDF["endereco"].apply(lambda x: logoddsPA[x])
address_features = trainDF["endereco"].apply(lambda x: logodds[x])
address_features.columns = ["logodds"+str(x) \
                for x in range(len(address_features.columns))]
feature_list = trainDF.columns.tolist()
feature_list.remove("categoria")
feature_list.remove("endereco")
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

model = LogisticRegression()
# model.fit(train_set,labels_train)

# print("all", \
#     log_loss(labels, model.predict_proba(data_set.as_matrix())))
# print("train", \
#     log_loss(labels_train, model.predict_proba(train_set.as_matrix())))
# print("test", \
#     log_loss(labels_test, model.predict_proba(test_set.as_matrix())))

# fit model to all training data

model.fit(data_set, labels)

# load the test data frame
testDF = pd.read_csv(os.getcwd() + os.sep + ".." + \
    os.sep + "datasets" + os.sep + "testFeaturizedAddress.csv")

# get the addresses present in the test dataset
test_addresses=sorted(testDF["endereco"].unique())
# get the number of unique addresses in the test set
test_A_counts=testDF.groupby("endereco").size()
# get the set of addresses that are exclusively in the test set
only_new=set(test_addresses+addresses)-set(addresses)
# get the set of addresses that are exclusively in the training set
only_old=set(test_addresses+addresses)-set(test_addresses)
# get the insertection between the addresses on the test and training sets
in_both=set(test_addresses).intersection(addresses)
# calculate log_odds for the address (in the test set)
for addr in only_new:
    PA=test_A_counts[addr]/float(len(testDF)+len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    logodds[addr].index=range(len(categories))
for addr in in_both:
    PA=(A_counts[addr]+test_A_counts[addr])/float(len(testDF)+len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
# set the logoddsPA in the data frame
testDF["logoddsPA"] = testDF["endereco"].apply(lambda x: logoddsPA[x])
test_address_features = testDF["endereco"].apply(lambda x: logodds[x])
test_address_features.columns = ["logodds"+str(x) \
                for x in range(len(test_address_features.columns))]
# remove some endereco field
test_feature_list = testDF.columns.tolist()
test_feature_list.remove("endereco")
test_feature_list.remove("Id")
test_data_set = testDF[test_feature_list].join(test_address_features.ix[:,:])

# get the probabilities for the test set
probDF = pd.DataFrame(model.predict_proba(test_data_set.as_matrix()),\
    columns=sorted(labels.unique()))

probDF.to_csv(os.getcwd() + os.sep + ".." + \
    os.sep + "datasets" + os.sep + "probability_result.csv",\
    index_label="Id", na_rep="0")
