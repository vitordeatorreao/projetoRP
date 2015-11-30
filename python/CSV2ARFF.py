import sys, os

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
    g = open(os.getcwd() + "\\..\\datasets\\trainFeaturized.arff", "w")
g.write(x+dat)
g.close()
