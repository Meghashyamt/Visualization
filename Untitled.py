#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel(r"/home/shyam/Downloads/Assignment/Dataset_Yield_Quality_Planting_Dates.xlsx")

# Display the first few rows of the DataFrame
print(data.head())

# Visualize the data
# Scatter plot for Planting Week vs Yield
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Planting Week', y='Yield', hue='Location', palette='viridis')
plt.title('Planting Week vs Yield')
plt.xlabel('Planting Week')
plt.ylabel('Yield')
plt.legend(title='Location')
plt.show()

# Bar plot for Hybrid
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Hybrid', palette='Set2')
plt.title('Count of Hybrids')
plt.xlabel('Hybrid')
plt.ylabel('Count')
plt.show()


# In[5]:


# Visualization for categorical data
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=data)
plt.title("Count of Locations")
plt.xlabel("Location")
plt.ylabel("Count")
plt.show()

# Visualization for ordinal data
plt.figure(figsize=(10, 6))
sns.boxplot(x='Hybrid', y='Yield', data=data)
plt.title("Boxplot of Yield by Hybrid Type")
plt.xlabel("Hybrid")
plt.ylabel("Yield")
plt.show()


# In[6]:


# Bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='Location', data=data)
plt.title('Bar Chart of Locations')
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Yield'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.show()

# Line graph
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Yield'], marker='o', linestyle='-')
plt.title('Line Graph of Yield Over Years')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.grid(True)
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
labels = data['Hybrid'].value_counts().index
sizes = data['Hybrid'].value_counts().values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Hybrid Types')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Yield', y='Quality', data=data)
plt.title('Scatter Plot of Yield vs Quality')
plt.xlabel('Yield')
plt.ylabel('Quality')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Location', y='Yield', data=data)
plt.title('Box Plot of Yield Across Locations')
plt.xlabel('Location')
plt.ylabel('Yield')
plt.show()


# In[9]:


# 1. Identifying and handling missing data
# Check for missing values
import numpy as np
import pandas as pd
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values by dropping rows with missing values
data = data.dropna()

# 2. Removing outliers (example using Z-score)
from scipy import stats
z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
filtered_entries = (z_scores < 3).all(axis=1)
data = data[filtered_entries]
print(data)

# 3. Data normalization and standardization
# Assuming 'Yield' and 'Quality' are the numerical features to be normalized/standardized
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Yield', 'Quality']] = scaler.fit_transform(data[['Yield', 'Quality']])
print(data)

# 4. Data aggregation and summarization
# Example: Group by 'Location' and calculate mean
summary_data = data.groupby('Location').mean()

print(summary_data.head())


# In[ ]:





# In[13]:


import plotly.express as px
import pandas as pd

# Assuming the dataset is stored in a CSV file named 'data.csv'
#sdata = pd.read_csv('data.csv')

# Creating an interactive scatter plot
fig = px.scatter(data, x='Yield', y='Quality', color='Location', size='Planting Week',
                 hover_data=['Year', 'Hybrid'],
                 title='Interactive Scatter Plot of Yield vs Quality')

# Adding interactive features such as hover information
fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

# Updating layout for better presentation
fig.update_layout(title='Interactive Scatter Plot of Yield vs Quality',
                  xaxis_title='Yield',
                  yaxis_title='Quality')

# Showing the interactive plot
fig.show()


# In[15]:


fig = px.histogram(data, x='Yield', nbins=20, title='Yield Distribution')
fig.show()


# In[16]:


fig = px.box(data, x='Location', y='Yield', points="all", title='Yield Distribution by Location')
fig.show()


# In[17]:


fig = px.bar(data, x='Location', y='Yield', title='Mean Yield by Location', barmode='group')
fig.show()


# In[18]:


fig = px.scatter_3d(data, x='Yield', y='Quality', z='Planting Week', color='Location')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




