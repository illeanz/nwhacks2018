import pymongo
import json
import pandas as pd

uri = 'mongodb://chickenlittle:butter@ds255797.mlab.com:55797/population_db'

count = 0
time_delay = 0.01
client = pymongo.MongoClient(uri)
db = client.get_database('population_db')

people = pd.read_csv('People.csv', header=0, index_col=0)
records = json.loads(people.T.to_json()).values()
db.population_db.insert_many(records)
