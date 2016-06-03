import pymongo

connection = pymongo.MongoClient('localhost',27017)
db = connection["allData"]
pageCollection = db["nb"]

pageCollection.drop()


