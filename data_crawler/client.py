import pymongo
import json

connection = pymongo.MongoClient('localhost',27017)
db = connection["allData"]
pageCollection = db["currDB"]

print str((pageCollection.count()))
totalExtScripts = 0


for page in pageCollection.find():
	for key in page.keys():
		if key != "text" and key != "scriptObjs" and key != "page_text":
			print str(key) + " = " + str(page[key])
