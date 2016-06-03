import pymongo
import json
from time import sleep
import json

connection = pymongo.MongoClient('localhost',27017)
db = connection["allData"]
pageCollection = db["currDB"]

#print str((pageCollection.count()))
totalExtScripts = 0

count = 0
for page in pageCollection.find():
	for key in page.keys():
		if key!="page_text" and key!="all_list_data":
			try:
				if (key=="scriptObjs"):
					for item in page[key]:
						#print "going to decode"
						decodedItem = json.loads(item)
						print "Hash " + decodedItem["textHash"]
				else:	
					print key +" = " + str(page[key])
				sleep(0.2)
			except:
				continue
	#if(count == 1):
		#break
	count += 1	
	print "\n\n"

		
