import pymongo
import json

connection = pymongo.MongoClient('localhost',27017)
db = connection["allData"]
pageCollection = db["nb"]

print str((pageCollection.count()))
totalExtScripts = 0


searchHash =  "2d6cfaa22ffbce65903f929c2d0263a40797a32a4670bd252160c0022b0173ab53266b73ab952730391401caaa847697061de05cf011077d2f11dfdaa2627093"


for item in pageCollection.find({},{"scriptObjs":1}):
	#each item represents a sciprtObj array

	#print str(type(item["scriptObjs"]))

	for scriptItem in item["scriptObjs"]:
		decodedItem = json.loads(scriptItem)
		for key in decodedItem.keys():
			if decodedItem[key]== searchHash:
				print "Found!"

		

	#decodedItem = json.loads(item["scriptObjs"])


"""
for page in pageCollection.find():
	#for key in page.keys():
	#	if page[key]!=None and key!="page_text" and key!="all_list_data":
	#		print "key: " + key + "   "+ str(page[key]);
	#print page["domain_name"]
	print page["url"]
	#print page["ranking"]
	#print page["url_len"]
	#print str(len(page["scriptObjs"]["allScripts"]))
	print
	#print page["num_iframes_present"]
	

	for item in page["scriptObjs"]:	
		decodedItem = json.loads(item)
		print decodedItem["text"]
		#print decodedItem["parent"]
	
		for key in decodedItem.keys():
			if key!= "text" and key!="textHash":	
				print "keys are " + str(key) + "value is " + str(decodedItem[key])
				pass
			if(len(decodedItem.keys())!=8):
				print "something wrong "
		
		#print item["textHash"]
		print
		print
	print totalExtScripts
"""
