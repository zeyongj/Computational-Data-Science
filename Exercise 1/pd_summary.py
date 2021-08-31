#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Part 0: Import Data.

# In[2]:


totals = pd.read_csv('totals.csv').set_index(keys=['name'])


# In[3]:


counts = pd.read_csv('counts.csv').set_index(keys=['name'])


# # Part 1: Which city had the lowest total precipitation over the year?

# In[4]:


city_total_precipitation = totals.sum(axis = 1) 


# In[5]:


lowest_precipitation_city = city_total_precipitation.idxmin(axis = 1)
# Adapted from https://www.geeksforgeeks.org/get-minimum-values-in-rows-or-columns-with-their-index-position-in-pandas-dataframe/


# In[6]:


# print("City with lowest total precipitation:")
# print(lowest_precipitation_city)


# # Part 2: Determine the average precipitation in these locations for each month.

# In[7]:


month_total_precipitation = totals.sum(axis = 0)


# In[8]:


month_total_observation = counts.sum(axis = 0)


# In[9]:


month_average_precipitation = month_total_precipitation / month_total_observation # According to the instrunction.


# In[10]:


# print("Average precipitation in each month:")
# print(month_average_precipitation) 


# # Part 3: Give the average precipitation (daily precipitation averaged over the month) for each city by printing the array.

# In[11]:


city_total_observation = counts.sum(axis = 1) 


# In[12]:


city_average_precipitation = city_total_precipitation / city_total_observation # According to the instrunction.


# In[13]:


# print("Average precipitation in each city:")
# print(city_average_precipitation) 


# # Part 4: Summary And Print

# In[14]:


print("City with lowest total precipitation:")
print(lowest_precipitation_city)
print("Average precipitation in each month:")
print(month_average_precipitation) 
print("Average precipitation in each city:")
print(city_average_precipitation) 

