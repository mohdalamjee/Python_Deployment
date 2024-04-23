import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# create a data-frame 
df = pd.read_excel('Vix historical data.xlsx')

numeric_data = df[['Open ', 'Close ']]

scaler = StandardScaler()
numeric_data_normalized = scaler.fit_transform(numeric_data)

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(numeric_data_normalized)
    wcss.append(km.inertia_)

X = numeric_data_normalized[:,:]
km = KMeans(n_clusters=6)
y_means = km.fit_predict(X)

df['Cluster'] = y_means

df.to_csv('final_report.csv', index=False)
