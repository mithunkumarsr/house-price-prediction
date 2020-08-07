import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

df = pd.read_csv(
    '/Users/mithunkumar/Desktop/HomePrices/homepricesMultivariate.csv')

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df.price)
#print(reg.predict([[3000, 3, 40]]))

# Saving model to disk
pickle.dump(reg, open('/Users/mithunkumar/Desktop/HomePrices/model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[15, 3000, 4]]))
