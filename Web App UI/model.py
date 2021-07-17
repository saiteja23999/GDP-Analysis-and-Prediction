import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
df = pd.read_csv("world.csv")
df.groupby('Region')[['GDP ($ per capita)','Literacy (%)','Agriculture']].median()
for col in df.columns.values:
    if df[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = df.groupby('Region')[col].median()
    for region in df['Region'].unique():
        df[col].loc[(df[col].isnull())&(df['Region']==region)] = guess_values[region]

der_df = df.drop(['Country','Region','Climate'],axis=1)
x= der_df.drop('GDP ($ per capita)',axis=1)
y= der_df['GDP ($ per capita)']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
print(xTest.shape)
regressor = RandomForestRegressor(n_estimators = 50,
                             max_depth = 6,
                             min_weight_fraction_leaf = 0.05,
                             max_features = 0.8,
                             random_state = 42)

#Fitting model with trainig data
regressor.fit(x, y)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
result = model.score(xTest, yTest)
x = ['20264082','7686850','2.6','0.34','3.98','4.69','100.0','565.5','6.55','0.04','93.41','12.14','7.51','0.038','0.262','0.7']
result1 = model.predict([x])
print(result1)


