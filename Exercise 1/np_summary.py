#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Part A: Which city had the lowest total precipitation over the year?

# In[2]:


data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']


# In[3]:


#totals


# In[4]:


#counts


# In[5]:


city_total_precipitation = np.sum(totals, axis = 1) # According to the instrunction.


# In[6]:


lowest_precipitation_city = np.argmin(city_total_precipitation)
# Adapted from https://numpy.org/doc/stable/reference/generated/numpy.argmin.html.


# In[7]:


# print("Row with lowest total precipitation:\n", lowest_precipitation_city) 


# # Part B: Determine the average precipitation in these locations for each month. 

# In[8]:


month_total_precipitation = np.sum(totals, axis = 0)


# In[9]:


month_total_observation = np.sum(counts, axis = 0)


# In[10]:


month_average_precipitation = month_total_precipitation / month_total_observation # According to the instrunction.


# In[11]:


# print("Average precipitation in each month:\n", month_average_precipitation) 


# # Part C: Give the average precipitation (daily precipitation averaged over the month) for each city by printing the array.

# In[12]:


city_total_observation = np.sum(counts, axis = 1)


# In[13]:


city_average_precipitation = city_total_precipitation / city_total_observation # According to the instrunction.


# In[14]:


# print("Average precipitation in each city:\n", city_average_precipitation) 


# # Part D: Calculate the total precipitation for each quarter in each city (i.e. the totals for each station across three-month groups). 

# In[15]:


number_of_rows, number_of_columns = totals.shape
# Adapted from https://stackoverflow.com/questions/18688948/numpy-how-do-i-find-total-rows-in-a-2d-array-and-total-column-in-a-1d-array.


# In[16]:


number_of_stations = number_of_rows


# In[17]:


quarter_reshape_precipitation = totals.reshape(4*number_of_stations,3) # According to the instrunction.


# In[18]:


quarter_sum_precipitation = np.sum(quarter_reshape_precipitation, axis = 1)


# In[19]:


quarter_total_precipitation = quarter_sum_precipitation.reshape(number_of_stations, 4)


# In[20]:


# print("Quarterly precipitation totals:\n", quarter_total_precipitation)


# # Part E: Summary And Print

# In[21]:


print("Row with lowest total precipitation:")
print(lowest_precipitation_city)
print("Average precipitation in each month:")
print(month_average_precipitation) 
print("Average precipitation in each city:")
print(city_average_precipitation) 
print("Quarterly precipitation totals:")
print(quarter_total_precipitation)


# In[ ]:




