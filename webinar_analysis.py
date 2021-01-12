#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:11:27 2021

@author: chaitanya
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pingouin as pg


data=pd.read_csv('Statistics from webinars - Sheet1.csv')
samp=data.copy(deep=True)

L=pd.crosstab(index=samp['Language'], columns='count')
# print(pd.crosstab(index=samp['Gestures'], columns=data['Rating']))

numerical_data=samp.select_dtypes(exclude=[object])
numerical_data=pd.DataFrame(numerical_data.drop(['Silence', 'Host talk', \
    'Host visual', 'Intonations', 'Likes', 'Dislikes', 'Duration',\
        'Host audio activity','Gest per video'], axis=1))
corr_mat=numerical_data.corr(method='pearson')
#plt.scatter(samp['Intonations per min'], samp['Rating'], c='r')
#plt.hist(samp['Rating'], color='green', edgecolor='white', bins=50)

count=[192, 23]
lan=['English', 'Russian']
index=np.arange(len(lan))
# plt.bar(index, count, color=['red', 'blue'])
# plt.xticks(index, lan)

# plt.show()

# sns.set(style='darkgrid')
# sns.regplot( samp['Rating'], samp['Host visual activity'])

# # sns.lmplot(x='Gestures', y='Rating', data=samp, hue='Language', legend=True, palette='Set1')

# sns.distplot(samp['Host visual activity'], bins=50)
# sns.countplot(samp['Language'])

# sns.countplot(x='Language', data=samp, hue='Mode')

# sns.boxplot(y=samp['Host visual activity'])

# sns.boxplot(y=samp['Rating'], x=samp['Language'], hue='Mode', data=samp)

# f, (ax_box, ax_hist)=plt.subplots(2, gridspec_kw={"height_ratios":(0.3, 0.7)})
# sns.boxplot(samp['Rating'], ax=ax_box)
# sns.distplot(samp['Rating'], ax=ax_hist)

# sns.pairplot(samp, kind='scatter', hue='Language')

# samp.insert(19, 'Satisfaction',"")

# i=0

# while i< len(samp['Rating']):
#     if samp['Rating'][i]>4.92:
#         samp['Satisfaction'][i]='Good'
#     elif (samp['Rating'][i]<4.66):
#         samp['Satisfaction'][i]='Poor'
#     else:
#         samp['Satisfaction'][i]='Average'
#     i=i+1
    
print(pg.corr(x=samp['Intonations per min'], y=samp['Rating']))
