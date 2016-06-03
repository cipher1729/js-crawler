import os
import subprocess
import urllib2, zipfile,cStringIO
import csv

from sys import argv



#strip out all the URLs and concatenate them into a string 
	#fetch the zip for the remote list of the alexa 1 million sites	
alexaUrl = "http://s3.amazonaws.com/alexa-static/top-1m.csv.zip"	
try:
	remoteZip = urllib2.urlopen(alexaUrl);
	readZipFile = cStringIO.StringIO(remoteZip.read()) 
	actualZip = zipfile.ZipFile(readZipFile);
	#read the actual file from the zip
	
	for fn in actualZip.namelist():
		urlsFile = actualZip.read(fn);
		
		"""
		#print fn;
		csvFile= csv.reader(urlsFile)	
		
		currUrl="";
		urlStart=-1;
		urlEnd=-1;
		listOfUrls=[];
		listOfUrlsInString="";
		
		#counter=0;
		#store all the URLs in a string 
		for row in csvFile:
			#counter+=1;
			#if(counter<20):
				#print row				
			if (len(row)==2):				
				currUrl+='%';
			elif(len(row)==0):
				currUrl+= '!';		
			else:
				currUrl+= row[0];	
	
		i=0;
		while(i<len(currUrl)):
			if (currUrl[i]=='%'):
				urlStart=i+1;
			if(currUrl[i]=='!'):
				urlEnd=i-1;
				listOfUrls.append("\""+currUrl[urlStart:urlEnd+1] + "\",");				
			i+=1;			
		"""
		"""i=0;
		while(i<20):
			print listOfUrls[i];
			i+=1;
		"""
		"""
		print len(listOfUrls);
		"""
		f= open("alexaUrls.txt","w")
		f.write(urlsFile)
		"""
		i=0;
		while(i<len(listOfUrls)):
			f.write(str(i)+" "+"http://www."+listOfUrls[i]);
			f.write("\n");
			listOfUrlsInString+= "http://www."+ listOfUrls[i];
			i+=1;
		"""
		f.close();
except urllib2.HTTPError:
	print "Something went wrong :("


