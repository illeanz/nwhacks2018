import pandas as pd
import gmaps
import json

gmaps.configure(api_key="AIzaSyBpofSmuGBGRurPJ4Pj1TitbwLKgQjMQiI") # Your Google API key

#Read in people's information

people = pd.read_csv('People.csv', header=0, index_col=1)
new_people = people[['Pos_Y', 'Classification']].copy()

#1. Generates JSON for heatmap from user info
x = new_people.reset_index().T.to_dict().values()
with open('heatmap.json', 'w+') as f:
    f.write(json.dumps(list(x)))