#!/usr/bin/env python
# coding: utf-8

# In[144]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[145]:


df = pd.read_csv('Workshop-5-dataset.zip', sep='\t',dtype=np.str)


# Printing the first five rows of the dataset.

# In[146]:


df.head()


# ### Number of rows and columns in the dataset

# In[147]:


df.shape


# We have 31941 rows and 44 columns

# In[148]:


STUDENT_NAME = 'AminaUmarFaragai'
STUDENT_NO = '5230'


# In[149]:


np.random.seed(int(STUDENT_NO))
unique_id = int('2' + STUDENT_NO)
rows = np.random.choice(df.index.values, unique_id)
data = df.loc[rows]


# In[150]:


file_name = STUDENT_NAME + "_" + STUDENT_NO + ".csv"
data.to_csv(file_name) 


# ### number of unique dates in the dataset
# In[151]:


#the unique dates in the dataset
len(data['Date'].unique())


# We have 305 unique dates.

# In[152]:


data['Date'].unique()


# In[153]:


data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour


# In[154]:


hour_hist = data.hist(column="Hour", bins=15, grid=False)
 
for ax in hour_hist.flatten():
    ax.set_xlabel("Hour")
    ax.set_ylabel("Frequency")
 


# In[155]:


data.info()


# In[156]:


data['Date'] = pd.to_datetime(data['Date'])


# In[157]:


pip install apyori


# In[158]:


# import apyori
from apyori import apriori


# In[159]:


data.head(1)


# In[160]:


items_df=data[data.columns[3:44]]


# In[161]:


items_df.head()


# In[162]:


baskets = items_df.T.apply(lambda x: x.dropna().tolist()).tolist()


# In[163]:


for i in baskets[:5]:
    print(i)


# In[164]:


association_rules = apriori(baskets, min_support=0.01, min_confidence=0.2, 
                            min_lift=3, min_length=2)
association_results = list(association_rules)


# In[165]:


print('Rules generated: ', len(association_results))


# In[166]:


print(association_results[0])


# In[167]:


print(association_results[1])


# In[168]:


print(association_results[1][0])


# In[169]:


def display_rules(association_results):
    for item in association_results:
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


# In[170]:


display_rules(association_results[:42])


# In[171]:


from collections import Counter
 
counter = Counter(baskets[0])
for i in baskets[1:]:
    if i != 'nan':
        counter.update(i)
 
del counter['nan']
mostcommon = counter.most_common(10)


# In[172]:


mostcommon


# In[173]:


item = [x[0] for x in mostcommon]
item_number = [x[1] for x in mostcommon]


# In[174]:


item

# Based on the top 10 association rule, none of the most common items are in top 10 of the association result displayed. This is because the most common items were mostly purchased as single items, and not in combination with other items. Although, the most common items had the highest number of sales in the shop, therefore, it could be suggested that the shelves in the shop be re-arranged such that, less comonly bought items be placed next to the most frequently purchased items on the shelves.
# 
# Another reason why these products were the most common, but not in top 10 association rule might be because these particular products are cheap in this shop, compared to other items, in different shops, so customers tend to buy only those items from this shop. Hence, running promotions on buying the less purchased items together with the most common items at a lesser price might increase the sales.

# ### Running the apriori algorithm with the following three different settings

# In[175]:


#Setting 1: Min Support = 0.015, Min Confidence = 0.7, Min Lift = 3

association_rules = apriori(baskets, min_support=0.015, min_confidence=0.7, 
                            min_lift=3, min_length=2)
association_results = list(association_rules)


# In[176]:


print('Rules generated: ', len(association_results))


# In[177]:


print(association_results[0])


# In[178]:


print(association_results[1])


# In[179]:


print(association_results[3])


# In[180]:


def display_rules(association_results):
    for item in association_results:
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


# In[181]:


display_rules(association_results[:4])


# In[182]:


#Setting 2: Min Support = 0.009, Min Confidence = 0.5, Min Lift = 3

association_rules = apriori(baskets, min_support=0.009, min_confidence=0.5, 
                            min_lift=3, min_length=2)
association_results = list(association_rules)


# In[183]:


print('Rules generated: ', len(association_results))


# In[184]:


print(association_results[0])


# In[185]:


def display_rules(association_results):
    for item in association_results:
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


# In[186]:


display_rules(association_results[:12])


# In[187]:


#Setting 3: Min Support = 0.015, Min Confidence = 0.5, Min Lift = 9
association_rules = apriori(baskets, min_support=0.015, min_confidence=0.5, 
                            min_lift=9, min_length=2)
association_results = list(association_rules)


# In[188]:


print('Rules generated: ', len(association_results))


# In[189]:


print(association_results[0])


# In[190]:


display_rules(association_results[:12])


# Calculating the number of rules you get for each setting and how the quality of the rules differ in each setting.
# 
# Setting 1: Min Support = 0.015, Min Confidence = 0.7, Min Lift = 3, rules generated = 4
# 
# Setting 2: Min Support = 0.009, Min Confidence = 0.5, Min Lift = 3, rules generated = 39
# 
# Setting 3: Min Support = 0.015, Min Confidence = 0.5, Min Lift = 9, rules generated = 9
# 
# In aprori, setting high parameters limit the size, number of rules, and strength of rules generated. This largely depends on what the organisation wants and the reasons behind the algorithm. If more vital rules are the focus, increasing the value of confidence will give the best result like in Setting 1, and reducing values give more rules like in Setting 2, but this has its limitations. 
# 
# 
# For Setting 1, the rules are strong, but some are repeated which makes them irrelevant, thus reducing the quality of rules generated using the first set.
# 
# For Setting 2, more extensive set of rules are generated.
# 
# For Setting 3, the rules are stronger than Setting 1, which is an improvement from Setting 1. The goal of association rule mining is to end up with rules that are direct.
# 
# Generally, rather than using thresholds to reduce the rules down to a smaller set, it is better for a more extensive set of rules to be returned so that there is a greater chance of generating relevant rules like in settings 2.
# For this store therefore, Setting 2 is recommended.

# ### Q6. Filter the transactions on the 'day' of the week or on the 'month' to perform analysis on either of them on two durations. Generate association rules to discover if there are significant differences in the buying behaviour between chosen durations, and, discuss if the rules are useful.

# In[191]:


data["Month"]= data['Date'].dt.month


# In[192]:


Month_hist = data.hist(column="Month", bins=12, grid=False)
 
for ax in Month_hist.flatten():
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency")


# Running comparison for 2 months July(summer) and December(winter).

# In[193]:


#January
month_jul= data.loc[data['Month']==7]


# In[194]:


#February
month_dec= data.loc[data['Month']==12]


# In[195]:


month_dec.head()


# In[196]:


items_df7=month_jul[month_jul.columns[3:44]]


# In[197]:


items_df12=month_dec[month_dec.columns[3:44]]


# In[198]:


baskets7 = items_df1.T.apply(lambda x: x.dropna().tolist()).tolist()


# In[199]:


baskets12 = items_df2.T.apply(lambda x: x.dropna().tolist()).tolist()


# In[200]:


print(len(baskets7))


# In[201]:


print(len(baskets12))


# In[202]:


for i in baskets7[:5]:
    print(i)


# In[203]:


for i in baskets12[:5]:
    print(i)


# In[204]:


association_rules7 = apriori(baskets7, min_support=0.01, min_confidence=0.2, 
                            min_lift=3, min_length=2)
association_results7 = list(association_rules7)


# In[205]:


association_rules12 = apriori(baskets12, min_support=0.01, min_confidence=0.2, 
                            min_lift=3, min_length=2)
association_results12 = list(association_rules12)


# In[206]:


print('Rules generated: ', len(association_results7))


# In[207]:


print('Rules generated: ', len(association_results12))


# In[208]:


print(association_results7[0])


# In[209]:


print(association_results12[0])


# In[210]:


def display_rules(association_results7):
    for item in association_results7:
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


# In[211]:


def display_rules(association_results12):
    for item in association_results12:
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        print("Support: " + str(item[1]))
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")


# In[212]:


display_rules(association_results7[:10])


# In[213]:


display_rules(association_results12[:10])


# In[214]:


from collections import Counter
 
counter = Counter(baskets7[0])
for i in baskets1[1:]:
    if i != 'nan':
        counter.update(i)
 
del counter['nan']
counter.most_common(10)


# In[215]:


from collections import Counter
 
counter = Counter(baskets12[0])
for i in baskets2[1:]:
    if i != 'nan':
        counter.update(i)
 
del counter['nan']
counter.most_common(10)







