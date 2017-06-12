import numpy as np
import pandas as pd
import re as re

train = pd.read_csv("C:\\Users\\bjzhaochen\Desktop\kaggle\\train.csv",header=0,dtype={'Age':np.float64})
test = pd.read_csv("C:\\Users\\bjzhaochen\Desktop\kaggle\\test.csv",header=0,dtype={'Age':np.float64})
full_data = [train,test]
# print(train.info())

# print(train[['Pclass','Survived']].groupby('Pclass',as_index=False).mean())
# print(train[['Sex','Survived']].groupby('Sex',as_index=False).mean())


#for dataset in full_data:
#   dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1
#print(train[['FamilySize','Survived']].groupby('FamilySize',as_index=False).mean())


# for dataset in full_data:
#     dataset['IsAlone'] = 0
#     dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())





