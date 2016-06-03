
import random
import getFeatures
from sys import argv
from bs4 import BeautifulSoup
import csv

def cleanString(dirtyString):
    output = "";
    for ch in dirtyString:
	if ord(ch)< 127:
		output += ch
    return output;

#print "Testing"
csvFile = open("data/all_bad_scripts_squidMal_new_features.csv","a")
csvWriter= csv.writer(csvFile)



#f= open("all_good_scripts_200.txt",'r');
f= open("/home/group9p1/javascripts_experiment/javascripts/ml/all_bad_scripts_squidMal_new.txt","r")
txt = f.read()
d=0

#jsFile = open("malJS.js","w")

#print str(len(f.read().split("\n"))


allVal=[]
totalScripts = len(txt.split("##########"))
#totalScripts = 100

invalid =0 
valid =0 
allJs = txt.split("##########")


for item in allJs:
	#for item in allJs[:len(allJs)-1]:
	#item = random.choice(allJs)
	currVal=[]
	#print str(len(item))
	#print
	#if d >=int(start) and d< int(end):
		
	if d>=0:
		#print item
		print
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

		#print ".....................................................SCRIPT COUNT"  + str(d+1)
		#skip this entry if could not be parsed
		if len(ret.keys())!=0:
			valid+=1
			for key in ret.keys():
				currVal.append(ret[key])
				#print "current key is " + str(key)  + " and value is "  + str(ret[key])
			#append a 1 for malicious
			currVal.append("0")
			
			#print "No of keys is " + str(len(ret.keys()))	
			if(currVal not in allVal and "-1" not in currVal):
				#allVal.append(currVal)
				csvWriter.writerow(currVal)
			#item = script.text
			#getFeatures.getFeatures(item)
			#jsFile.write(it
		else:
			invalid+=1;
	#print
	d+=1
	

#print "TOtal no of js" + str(totalScripts)
#print "No of failures"  + str(invalid)
#print "success " + str(valid)



f.close()
csvFile.close()
