#!/usr/bin/env python
# coding: utf-8

# # Categorical Data Plots
# 
# Now let's discuss using seaborn to plot categorical data! There are a few main plot types for this:
# 
# * factorplot
# * boxplot
# * barplot
# * countplot
# 
# Let's go through examples of each!

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


sns.get_dataset_names()


# In[3]:


tips = sns.load_dataset('tips')
tips.head()


# ## barplot and countplot
# 
# These very similar plots allow you to get aggregate data off a categorical feature in your data. **barplot** is a general plot that allows you to aggregate the categorical data based off some function, by default the mean:

# In[5]:



sns.barplot(x='time',y='total_bill',data=tips);


# In[6]:


sns.barplot(x='time',y='total_bill',data=tips,ci=0);   #  ci is confidence interval


# In[7]:


sns.barplot(x='sex',y='total_bill',data=tips, hue='time', palette='rainbow')
plt.legend(bbox_to_anchor=(1.3,1));


# ### countplot
# 
# This is essentially the same as barplot except the estimator is explicitly counting the number of occurrences. Which is why we only pass the x value:

# In[7]:


sns.countplot(x='time',data=tips)


# In[5]:


sns.countplot(x='sex',hue='smoker',data=tips, palette='BuGn');


# ## boxplot 
# 
# boxplots are used to shown the distribution of categorical data. A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.

# In[8]:


sns.boxplot(x="time", y="total_bill", data=tips,palette='rainbow')


# In[11]:


# Can do entire dataframe with orient='h'
sns.boxplot(data=tips,palette='rainbow',orient='h')


# In[6]:


sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")


# ## factorplot
# 
# factorplot is the most general form of a categorical plot. It can take in a **kind** parameter to adjust the plot type:

# In[8]:


sns.factorplot(x='day',y='total_bill',data=tips,kind='bar')


# In[8]:


sns.catplot(x='day',data=tips,kind='count')

