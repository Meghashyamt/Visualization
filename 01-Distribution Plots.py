#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Distribution Plots
# 
# Let's discuss some plots that allow us to visualize the distribution of a data set. These plots are:
# 
# * distplot
# * jointplot
# * pairplot
# 

# ___
# ## Imports

# In[2]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.get_dataset_names() # To view the inbuilt dataset names in seaborn library


# In[ ]:





# ## Data
# Seaborn comes with built-in data sets!

# In[3]:


tips = sns.load_dataset('tips')


# In[8]:


tips.head()


# In[9]:


import pandas as pd


# To view the entire dataset

# In[11]:


pd.set_option("display.max_rows", 244, "display.max_columns", 7)


# In[13]:


tips.shape


# In[14]:


tips


# ## distplot
# 
# The distplot shows the distribution of a univariate set of observations.

# In[6]:


sns.distplot(tips['total_bill']);


# To remove the kde layer and just have the histogram use:

# In[7]:


sns.distplot(tips['total_bill'],kde =False);


# 

# In[22]:


sns.distplot(tips['total_bill'],kde_kws={"color": "k", "lw": 3, "label": "KDE",'ls':'--'}, hist_kws={"alpha": 0.5, "color": "g",'edgecolor':'black'});   #kde kernel density estimator  histogram values


# ## jointplot
# 
# jointplot() allows you to basically match up two distplots for bivariate data. With your choice of what **kind** parameter to compare with: 
# * “scatter” 
# * “reg” 
# 

# In[24]:


sns.set(style="dark")
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter');


# In[28]:


sns.set(style="white")


# In[32]:


plot=sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')# Regression
plot.ax_marg_x.set_xlim(0, 40)
plot.ax_marg_y.set_ylim(0, 10);


# In[25]:





# ## pairplot
# 
# pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns) and supports a color hue argument (for categorical columns). 

# In[33]:


sns.pairplot(tips);


# In[12]:


sns.pairplot(tips,hue='sex',palette='coolwarm')


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


sns.pairplot(tips,hue='day',palette='bright');


# In[11]:


c=['blue','yellow']


# In[12]:


sns.pairplot(tips,hue='sex',palette=c)


# In[8]:


sns.pairplot(tips,vars=['total_bill','tip'],hue='sex');

