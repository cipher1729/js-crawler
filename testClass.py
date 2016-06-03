
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

print "Testing"
csvFile = open("data/small.csv","a")
csvWriter= csv.writer(csvFile)

passedScriptsFile = open("passedScriptsFile.txt","w")
failedScriptsFile = open("failedScriptsFile.txt","w")


f= open("ADD-3",'r');
#f= open("SUPERBAD1.txt","r")
txt = f.read()
d=0

#jsFile = open("malJS.js","w")

#print str(len(f.read().split("\n"))


allVal=[]
totalScripts = len(txt.split("###############"))
#totalScripts = 100

tempList= []
tempList.append(totalScripts-1)
tempList.append(28)

allVal.append(tempList)

invalid =0 
valid =0 
allJs = txt.split("###############")


for item in allJs[:len(allJs)-1]:
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

		print ".....................................................SCRIPT COUNT"  + str(d+1)
		#skip this entry if could not be parsed
		if len(ret.keys())!=0:
			valid+=1
			for key in ret.keys():
				currVal.append(ret[key])
				#print "current key is " + str(key)  + " and value is "  + str(ret[key])
			#append a 1 for malicious
			currVal.append("1")
			
			#print "No of keys is " + str(len(ret.keys()))	
			if(currVal not in allVal and "-1" not in currVal):
				#allVal.append(currVal)
				csvWriter.writerow(currVal)
				#passedScriptsFile.write(cleanString(item)+ "\n")
				#passedScriptsFile.write("##########+\n")
			else:
				#failedScriptsFile.write(cleanString(item)+ "\n")
				#failedScriptsFile.write("##########+\n")
				print cleanString(item)
			
			#item = script.text
			#getFeatures.getFeatures(item)
			#jsFile.write(it
		else:
			invalid+=1;
	#print
	d+=1


passedScriptsFile.close()
failedScriptsFile.close()
	
"""
f.close()
f= open("all_good_scripts_1.txt","r")
txt = f.read()

d=0
for item in txt.split("##########"):
	currVal=[]
	#print str(len(item))
	#print
	#if d >=int(start) and d< int(end):
		
	if d<1000:
		#print item
		print
		soup = BeautifulSoup(item)
		script = soup.find("script")
		ret ={}
		#condition for top level script
		if script is not None:
			if script.parent.name == "head"	:
				actItem = script.text
				ret = getFeatures.getFeatures(actItem)
			else:
				ret = getFeatures.getFeatures(item)
		else:
			ret = getFeatures.getFeatures(item)

		print ".....................................................SCRIPT COUNT"  + str(d+1)
		#skip this entry if could not be parsed
		if len(ret.keys())!=0:
			valid+=1
			for key in ret.keys():
				currVal.append(ret[key])
				print "current key is " + str(key)  + " and value is "  + str(ret[key])
			#append a 1 for malicious
			currVal.append("0")
			
			print "No of keys is " + str(len(ret.keys()))	
			allVal.append(currVal)
			#item = script.text
			#getFeatures.getFeatures(item)
			#jsFile.write(it
		else:
			invalid+=1;
	#print
	d+=1
	
"""

print "TOtal no of js" + str(totalScripts)
print "No of failures"  + str(invalid)
print "success " + str(valid)
#csvWriter.writerows(allVal);
#print str(len(txt.split("###############")))
#jsFile.close()
f.close()
csvFile.close()
