import pymongo

uri = 'mongodb://chickenlittle:butter@ds255797.mlab.com:55797/population_db'

client = pymongo.MongoClient(uri)
db = client.get_database('population_db')
data = db['population_db']
cursor = data.find({})

for document in cursor:
    print(document)

client.close