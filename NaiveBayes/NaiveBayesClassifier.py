# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:35:42 2017

@author: Abdullah
"""

import numpy as np
import pandas as pd

classProb = {}
featureProb_no = {}
featureProb_yes = {}

#Read two files - training and query files
df1 = pd.read_csv("tennis-training.txt")
df2 = pd.read_csv("tennis-query.txt", header=None)
#print(df1)
#print(df1.shape)

#Calculate class probabilties
#Divide the table for Play = No and Play = Yes
df_no = df1.loc[df1['Play'] == 'No']
df_yes = df1.loc[df1['Play'] == 'Yes']
no_nos = df_no.shape[0]
no_yes =  df_yes.shape[0]

classProb['No'] = no_nos / (no_nos + no_yes)
classProb['Yes'] = no_yes / (no_nos + no_yes)

print("The Trained Probabilities:")
print(classProb)

#Calculate conditional feature probabilities when Class = No
column_names = list(df_no)
#Remove column 'Play'
column_names = column_names[0:4]
for col in column_names:
    tot = 0
    #What all values a column/feature can take
    vals = df_no[col].unique()
    for val in vals:
        df = df_no[df_no[col] == val]
        featureProb_no[val] = df.shape[0]
        tot = tot + df.shape[0]
    #Convert number to actual probabailities 
    for val in vals:
        featureProb_no[val] = featureProb_no[val]/tot
 
print(featureProb_no)


#Calculate conditional feature probabilities when Class = Yes
column_names = list(df_yes)
#Remove column 'Play'
column_names = column_names[0:4]
for col in column_names:
    tot = 0
    #What all values a column/feature can take
    vals = df_yes[col].unique()
    for val in vals:
        df = df_yes[df_yes[col] == val]
        featureProb_yes[val] = df.shape[0]
        tot = tot + df.shape[0]
    #Convert number to actual probabailities 
    for val in vals:
        featureProb_yes[val] = featureProb_yes[val]/tot
 
print(featureProb_yes)


#We have all the probabilities
#Now classify using the input queries
print("\nThe Naive Bayes Classifier Results:")

n = df2.shape[0]
for i in range(n):
    row = df2[i:i+1]
    m=row.shape[1]

    #Calculate probabilites for class = No
    prob_no = 1
    prob_no = prob_no * classProb['No']
    for j in range(m):
        feature = row.iloc[0,j]
        prob_no = prob_no * featureProb_no[feature]

    #Calculate probabilites for class = Yes
    prob_yes = 1
    prob_yes = prob_yes * classProb['Yes']
    for j in range(m):
        feature = row.iloc[0,j]
        prob_yes = prob_yes * featureProb_yes[feature]

    #Output the actual class
    if (prob_yes >= prob_no):
        print("Yes")
    else:
        print("No")
        

        
