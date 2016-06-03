import getFeatures
from sys import argv
from bs4 import BeautifulSoup
import csv
import uuid
import virustotal2
import os 
import json 

def cleanString(dirtyString):
    output = "";
    for ch in dirtyString:
	if ord(ch)< 127:
		output += ch
    return output;


pass_count = 0
print "Testing"
script = argv
f= open("db/MalJs_3.txt",'r');
csvFile = open("MalJs_3.csv","w")
csvWriter= csv.writer(csvFile)
txt = f.read()
d=0

#jsFile = open("malJS.js","w")

#print str(len(f.read().split("\n"))


allVal=[]
api_key = "f6dd1ee9e7f17d55b70f47dfb993c59d873a0a3e8703f8eae143fd5fd60049a6"
if(os.path.exists('repo')):
    pass
else:
    os.mkdir('repo')
v = virustotal2.VirusTotal2(api_key)
for item in txt.split("###############"): ###############
    currVal=[] 
    #print str(len(item))
    #print
    if d >0:
        #print item
	soup = BeautifulSoup(item)
	script = soup.find("script")
	ret ={}
	#condition for top level script
	if script is not None:
	    if script.parent.name == "head"	:
	        actItem = script.text
                ret = getFeatures.getFeatures(cleanString(actItem))
            else:
                ret = getFeatures.getFeatures(cleanString(item))
        else:
	    ret = getFeatures.getFeatures(cleanString(item))

	print "................................................SCRIPT COUNT"  + str(d+1)

        u = str(uuid.uuid1())
        handle = open('./repo/%s'%u, 'w') 
        handle.write(item) 
        handle.close()
        currVal.append(u)
        try:
            report = v.retrieve('./repo/%s'%(u),raw ='True')

        except TypeError:
            print "Retrieve failed : ",u 
        try:
            repdict = json.loads(str(report))

        except ValueError:
            print "Json load failed"
            continue

        if(repdict['verbose_msg'] == 'Scan finished, information embedded'):
            currVal.append(repdict['positives'])
            currVal.append(repdict['total'])
            for scan in repdict['scans']:
                if(repdict['scans'][scan]['detected'] == True ):
                    currVal.append(repdict['scans'][scan]['result'])
                    break 

	if len(ret.keys())!=0:
            for key in ret.keys():
                currVal.append(ret[key])
                print "current key is " + str(key)  + " and value is "  + str(ret[key])
			#append a 1 for malicious
	    currVal.append("1")
            print "No of keys is " + str(len(ret.keys()))
            pass_count = pass_count + 1 
            if currVal not in allVal and "-1" not in currVal:
		allVal.append(currVal)
    d+=1
csvWriter.writerows(allVal);
f.close()
csvFile.close()
print " TOTAL SCRIPTS PARSED ----", pass_count 
