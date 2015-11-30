import sys, os
import pandas as pd
import numpy as np
from copy import deepcopy

if len(sys.argv) > 1:
    f = open(sys.argv[1])
else:
    f = open(os.getcwd() + "\\..\\datasets\\trainFeaturizedAddress.csv")
lines = f.read().split("\n")
data = []
for line in lines:
    if line == "":
        continue
    l = line.split(",")
    ds = []
    for attr in l:
        d = attr
        try:
            d = int(attr)
        except ValueError:
            try:
                a = attr.replace(".", ".")
                d = float(a)
            except ValueError:
                pass
        ds += [d]
    data += [ds]

f.close()

data = data[1:] #eliminate header

trainDF = pd.read_csv(os.getcwd() + "\\..\\datasets\\trainFeaturizedAddress.csv")
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
default_logodds=np.log(C_counts/len(trainDF))-np.log(1.0-C_counts/float(len(trainDF)))
for addr in addresses:
    PA=A_counts[addr]/float(len(trainDF))
    logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    logodds[addr]=deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
            PA=A_C_counts[addr][cat]/float(A_counts[addr])
            logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
    logodds[addr]=pd.Series(logodds[addr])
    logodds[addr].index=range(len(categories))


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
@ATTRIBUTE log_odds_0 NUMERIC
@ATTRIBUTE log_odds_1 NUMERIC
@ATTRIBUTE log_odds_2 NUMERIC
@ATTRIBUTE log_odds_3 NUMERIC
@ATTRIBUTE log_odds_4 NUMERIC
@ATTRIBUTE log_odds_5 NUMERIC
@ATTRIBUTE log_odds_6 NUMERIC
@ATTRIBUTE log_odds_7 NUMERIC
@ATTRIBUTE log_odds_8 NUMERIC
@ATTRIBUTE log_odds_9 NUMERIC
@ATTRIBUTE log_odds_10 NUMERIC
@ATTRIBUTE log_odds_11 NUMERIC
@ATTRIBUTE log_odds_12 NUMERIC
@ATTRIBUTE log_odds_13 NUMERIC
@ATTRIBUTE log_odds_14 NUMERIC
@ATTRIBUTE log_odds_15 NUMERIC
@ATTRIBUTE log_odds_16 NUMERIC
@ATTRIBUTE log_odds_17 NUMERIC
@ATTRIBUTE log_odds_18 NUMERIC
@ATTRIBUTE log_odds_19 NUMERIC
@ATTRIBUTE log_odds_20 NUMERIC
@ATTRIBUTE log_odds_21 NUMERIC
@ATTRIBUTE log_odds_22 NUMERIC
@ATTRIBUTE log_odds_23 NUMERIC
@ATTRIBUTE log_odds_24 NUMERIC
@ATTRIBUTE log_odds_25 NUMERIC
@ATTRIBUTE log_odds_26 NUMERIC
@ATTRIBUTE log_odds_27 NUMERIC
@ATTRIBUTE log_odds_28 NUMERIC
@ATTRIBUTE log_odds_29 NUMERIC
@ATTRIBUTE log_odds_30 NUMERIC
@ATTRIBUTE log_odds_31 NUMERIC
@ATTRIBUTE log_odds_32 NUMERIC
@ATTRIBUTE log_odds_33 NUMERIC
@ATTRIBUTE log_odds_34 NUMERIC
@ATTRIBUTE log_odds_35 NUMERIC
@ATTRIBUTE log_odds_36 NUMERIC
@ATTRIBUTE log_odds_37 NUMERIC
@ATTRIBUTE log_odds_38 NUMERIC
"""

x += """@ATTRIBUTE category {\"ARSON\",\"ASSAULT\",\"BAD CHECKS\",\"BRIBERY\",\"BURGLARY\",\"DISORDERLY CONDUCT\",\"DRIVING UNDER THE INFLUENCE\",\"DRUG/NARCOTIC\",\"DRUNKENNESS\",\"EMBEZZLEMENT\",\"EXTORTION\",\"FAMILY OFFENSES\",\"FORGERY/COUNTERFEITING\",\"FRAUD\",\"GAMBLING\",\"KIDNAPPING\",\"LARCENY/THEFT\",\"LIQUOR LAWS\",\"LOITERING\",\"MISSING PERSON\",\"NON-CRIMINAL\",\"OTHER OFFENSES\",\"PORNOGRAPHY/OBSCENE MAT\",\"PROSTITUTION\",\"RECOVERED VEHICLE\",\"ROBBERY\",\"RUNAWAY\",\"SECONDARY CODES\",\"SEX OFFENSES FORCIBLE\",\"SEX OFFENSES NON FORCIBLE\",\"STOLEN PROPERTY\",\"SUICIDE\",\"SUSPICIOUS OCC\",\"TREA\",\"TRESPASS\",\"VANDALISM\",\"VEHICLE THEFT\",\"WARRANTS\",\"WEAPON LAWS\"}

@DATA"""

for l in range(len(data)):
    addr = data[l][len(data[l])-2]
    rest = data[l][len(data[l])-1:]
    data[l] = data[l][:len(data[l])-2]
    for i in logodds[addr]:
        data[l] += [i]
    data[l] += rest

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
    f = open(sys.argv[2])
else:
    g = open(os.getcwd() + "\\..\\datasets\\trainFeaturizedLogOdds.arff", "w")
g.write(x+dat)
g.close()
