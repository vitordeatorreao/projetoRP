###############################################################################
# CSV2ARFF_Balance.py
###############################################################################
# Converts a CrimesSF featurized csv file into a featurized arff file
# so it can be used on WEKA
###############################################################################
# usage:
# > python CSV2ARFF_Balance.py <CSV_FILEPATH> <ARFF_FILEPATH> <MAX_FREQUENCY>
###############################################################################

import sys, os, math
import statistics as st
import numpy as np
from random import shuffle

def getAttr(data, index):
    d = []
    for i in range(0, len(data)):
        d += [data[i][index]]
    return d

def copy_column(data, j, col, fromFloatToInt=True):
    for l in range(len(data)):
        data[l][j] = col[l]

if len(sys.argv) > 1:
    f = open(sys.argv[1])
else:
    f = open(os.getcwd() + "\\..\\datasets\\trainFeaturized.csv")
lines = f.read().split("\n")
data = []
for line in lines:
    if line == "":
        continue
    l = line.split(";")
    ds = []
    for attr in l:
        d = attr
        try:
            d = int(attr)
        except ValueError:
            try:
                a = attr.replace(",", ".")
                d = float(a)
            except ValueError:
                pass
        ds += [d]
    data += [ds]

f.close()

x = """@RELATION crimesSF

@ATTRIBUTE latitude NUMERIC
@ATTRIBUTE longitude NUMERIC
@ATTRIBUTE hour NUMERIC
@ATTRIBUTE minute NUMERIC
@ATTRIBUTE day NUMERIC
@ATTRIBUTE month NUMERIC
@ATTRIBUTE year NUMERIC
@ATTRIBUTE street NUMERIC
@ATTRIBUTE intersection NUMERIC
@ATTRIBUTE pd_bayview NUMERIC
@ATTRIBUTE pd_central NUMERIC
@ATTRIBUTE pd_ingleside NUMERIC
@ATTRIBUTE pd_mission NUMERIC
@ATTRIBUTE pd_northern NUMERIC
@ATTRIBUTE pd_park NUMERIC
@ATTRIBUTE pd_richmond NUMERIC
@ATTRIBUTE pd_southern NUMERIC
@ATTRIBUTE pd_taraval NUMERIC
@ATTRIBUTE pd_tenderlion NUMERIC
@ATTRIBUTE sunday NUMERIC
@ATTRIBUTE monday NUMERIC
@ATTRIBUTE tuesday NUMERIC
@ATTRIBUTE wednesday NUMERIC
@ATTRIBUTE thrusday NUMERIC
@ATTRIBUTE friday NUMERIC
@ATTRIBUTE saturday NUMERIC
@ATTRIBUTE isAwake NUMERIC
@ATTRIBUTE summer NUMERIC
@ATTRIBUTE winter NUMERIC
@ATTRIBUTE autumn NUMERIC
@ATTRIBUTE spring NUMERIC
"""

x += """@ATTRIBUTE category {\"ARSON\",\"ASSAULT\",\"BAD CHECKS\",\"BRIBERY\",\"BURGLARY\",\"DISORDERLY CONDUCT\",\"DRIVING UNDER THE INFLUENCE\",\"DRUG/NARCOTIC\",\"DRUNKENNESS\",\"EMBEZZLEMENT\",\"EXTORTION\",\"FAMILY OFFENSES\",\"FORGERY/COUNTERFEITING\",\"FRAUD\",\"GAMBLING\",\"KIDNAPPING\",\"LARCENY/THEFT\",\"LIQUOR LAWS\",\"LOITERING\",\"MISSING PERSON\",\"NON-CRIMINAL\",\"OTHER OFFENSES\",\"PORNOGRAPHY/OBSCENE MAT\",\"PROSTITUTION\",\"RECOVERED VEHICLE\",\"ROBBERY\",\"RUNAWAY\",\"SECONDARY CODES\",\"SEX OFFENSES FORCIBLE\",\"SEX OFFENSES NON FORCIBLE\",\"STOLEN PROPERTY\",\"SUICIDE\",\"SUSPICIOUS OCC\",\"TREA\",\"TRESPASS\",\"VANDALISM\",\"VEHICLE THEFT\",\"WARRANTS\",\"WEAPON LAWS\"}

@DATA"""

# balance
if len(sys.argv) > 3:
    MAX_FREQUENCY = int(sys.argv[3])
else:
    MAX_FREQUENCY = 5000
classes = {}
clss = set()
balanced_data = []
for instance in data:
    clss.add(instance[-1])

for cls in clss:
    classes[cls] = 0

# crop highly frequent classes
shuffle(data) # make it random
for instance in data:
    if classes[instance[-1]] < MAX_FREQUENCY:
        balanced_data += [instance]
        classes[instance[-1]] += 1

# increase frequency of classes below frequency upper limit
for cls in clss:
    if classes[cls] >= MAX_FREQUENCY:
        continue
    # capture instances from that class
    instances = []
    for instance in balanced_data:
        if instance[-1] == cls:
            instances += [instance]
    # create new instances
    new_instances = np.array([[0] * len(instances[0])] * (MAX_FREQUENCY - len(instances)), dtype=np.float)
    for i in range(len(instances[0])-1):
        #sys.stdout.write("\rCopying atribute" + str(i) + "/" + str(len(instances[0])-2) + "of class "+ cls +"                                                                                                ")
        # create new atribute column with Gaussian Noise
        atrib = getAttr(instances, i)
        mean = st.mean(atrib)
        stdv = st.stdev(atrib)
        if mean == 0.0 and stdv == 0.0:
            column = np.zeros(MAX_FREQUENCY - len(instances))
        else:
            column = np.random.normal(mean, stdv, MAX_FREQUENCY - len(instances))
        new_instances[:,i] = column
    # set the new_instances' class
    adding_instances = []
    for instance in new_instances:
        instance = instance.tolist()
        adding_instances += [instance]
    for instance in adding_instances:
        for j in range(len(instance)):
            if j == len(instance)-1:
                instance[j] = cls
            elif j != 0 and j != 1:
                instance[j] = math.ceil(instance[j]) #if math.floor(instance[j]) > 0 else math.floor(instance[j])*-1 
    # add new instances to data set
    balanced_data += adding_instances
# set dataset to be the balanced newly created one
data = balanced_data

dat = "\n"
linec = 0
sys.stdout.write("\n")
for line in data:
    i = 0
    for i in range(0, len(line)-1):
        if type(line[i]) != str:
            dat += str(line[i]) + "," 
        else:
            dat += "\"" + line[i] + "\"," 
    i += 1
    if type(line[i]) != str:
        dat += str(line[i]) + "\n"
    else:
        dat += "\"" + line[i] + "\"\n" 
    linec += 1
    sys.stdout.write("\r" + str(linec) + "/" + str(len(data)) + "                                    ")

if len(sys.argv) > 2:
    g = open(sys.argv[2], "w")
else:
    g = open(os.getcwd() + "\\..\\datasets\\trainFeaturizedBalanced5.arff", "w")
g.write(x+dat)
g.close()
