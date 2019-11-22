#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import numpy as np
import pandas as pd
from math import sqrt

def read_file_into_df(dir_):
    """Returns a DataFrame with Testing Data"""
    l1=[]
    with open(dir_,"r", encoding="latin1") as file:
        for line in file:
            l1.append(list(map(float,line.rstrip().split(','))))
    testing_df = pd.DataFrame(l1, columns = ['MovieID','UserID','Rating']).sort_values(by=['UserID','MovieID'])
    return testing_df
    
def read_file_into_list(directory):
    """Returns a DataFrame with Training Data, User & Movie dictionaries and a DataFrame with Mean vote of the user"""
    #l2=np.loadtxt("netflix\TestingRatings.txt", delimiter=',', dtype=int)
    #Alternate method below which is 3 times faster
    l1=[]
    userDict={}
    movieDict={}
    with open(directory,"r", encoding="latin1") as file:
        for line in file:
            l1.append(list(map(float,line.rstrip().split(','))))
            if l1[-1][1] in userDict:
                userDict[l1[-1][1]].append(l1[-1][0])
            else:
                userDict[l1[-1][1]]=[l1[-1][0]]
                
            if l1[-1][0] in movieDict:
                movieDict[l1[-1][0]].append(l1[-1][1])
            else:
                movieDict[l1[-1][0]]=[l1[-1][1]]

    mean_vote_of_user = pd.DataFrame(l1, columns = ['MovieID','UserID','Rating']).groupby(['UserID'], as_index=False).Rating.mean()
    mean_vote_of_user.columns = ['UserID', 'MeanRating']
    df_training_list = pd.DataFrame(l1, columns = ['MovieID','UserID','Rating'])
    merged_df = pd.merge(df_training_list, mean_vote_of_user, on='UserID')
    merged_df["Diff"] = merged_df["Rating"]-merged_df["MeanRating"]
    return merged_df.drop(columns=['Rating', 'MeanRating']), userDict, movieDict, mean_vote_of_user

def get_error(testing_df):
    """Calculates and returns Mean Absolute Error as MAE and Root Mean Squared Error as RMSE"""
    MAE = (testing_df['Rating']-testing_df['Prediction']).abs().sum()/testing_df.shape[0]
    RMSE = sqrt(((testing_df['Rating']-testing_df['Prediction'])**2).sum()/testing_df.shape[0])
    return MAE, RMSE

if __name__ == '__main__':
    import numpy as np
    import os
    import time
    import sys
    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    
    #train_dir = "netflix\TrainingRatings.txt"
    #test_dir = "netflix\TestingRatings.txt"
    
    t0 = time.time()
        
    training_df, trainuserDict, trainmovieDict, user_mean = read_file_into_list(train_dir)
    training_df = training_df.sort_values(by=['UserID','MovieID'])
    testing_df = read_file_into_df(test_dir)
    testing_df['Prediction'] = np.zeros(testing_df.shape[0])       #Enter default 0 value in all predictions
    
    print("Total users:",str(len(set(testing_df['UserID']))))
    count = 0
    #Run until all all predictions are updated in testing dataframe
    while(len(testing_df[testing_df['Prediction']==0]) > 0):
        user = (testing_df[testing_df['Prediction']==0].iloc[0])['UserID']
        merged_df = pd.merge(training_df[training_df['UserID']==user], training_df[training_df['MovieID'].isin(trainuserDict[user])], on='MovieID')
        merged_df = merged_df[merged_df['UserID_x']!=merged_df['UserID_y']]
        merged_df["Diff_x2"] = merged_df['Diff_x']**2
        merged_df["Diff_y2"] = merged_df['Diff_y']**2
        merged_df["Diff_xy"] = merged_df['Diff_x']*merged_df['Diff_y']
        merged_df = merged_df.drop(columns=['Diff_x','Diff_y','MovieID'])
        merged_df = merged_df.groupby(['UserID_x','UserID_y'], as_index=False).sum()
        merged_df["Diff_x2y2"] = merged_df['Diff_x2']*merged_df['Diff_y2']
        merged_df['w'] = merged_df['Diff_xy']/np.sqrt(merged_df['Diff_x2']*merged_df['Diff_y2'])
        merged_df = merged_df.drop(columns=['Diff_x2','Diff_y2','Diff_xy','Diff_x2y2'])
        merged_df = (pd.merge(pd.merge(testing_df, merged_df, left_on='UserID', right_on='UserID_x'), training_df, left_on=['MovieID', 'UserID_y'], right_on=['MovieID', 'UserID']))
        merged_df['wdiff'] = merged_df['w']*merged_df['Diff']
        merged_df = merged_df.drop(columns=['UserID_y','Diff','Rating','Prediction','UserID_x'])
        merged_df['w'] = np.absolute(merged_df['w'])
        merged_df = merged_df.groupby(['MovieID'], as_index=False).sum()
        testing_df.loc[testing_df['UserID']==user, 'Prediction'] = ((merged_df['wdiff']/merged_df['w'])+float(user_mean[user_mean['UserID']==user]['MeanRating'])).tolist()  
        count += 1
        print("Iteration:", count, "UserID:", int(user),"Time Elapsed:", time.time()-t0)
                
    MAE, RMSE = get_error(testing_df)
    print("Mean Absolute Error:", str(MAE), "& Root Mean Squared Error:", str(RMSE))
    print("Total Time Taken:", time.time()-t0)