import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os  # Import the os module to handle file paths

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Create a data-frame
df = pd.read_excel(os.path.join(script_dir, 'Vix historical data.xlsx'))

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

# Save the CSV file in the same directory
output_file = os.path.join(script_dir, 'final_report.csv')
df.to_csv(output_file, index=False)
