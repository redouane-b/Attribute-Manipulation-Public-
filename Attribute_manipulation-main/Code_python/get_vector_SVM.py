
from unittest import skip
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_vdvae
import pickle
from sklearn import linear_model,svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

#level of latent variables
data=np.load("Code_python/data/data_latents_for_5.npy",allow_pickle=True)
print(len(data))

X=[]
Y=[]

for i,elem in enumerate(data):
    x,label=elem

    if label["age_group"]=='0-2' or label["age_group"]=='3-6':
        X.append(x)
        Y.append(0)

    if label["age_group"]=='7-9' or label["age_group"]=='10-14':
        X.append(x)
        Y.append(1)
        
    if label["age_group"]=='15-19' or label["age_group"]=='20-29' or label["age_group"]=='30-39' or label["age_group"]=='40-49':
        X.append(x)
        Y.append(2)

    if label["age_group"]=='50-69' or label["age_group"]=='70-120':
        X.append(x)
        Y.append(3)



for k in range(6):

  coord=[]

  for x1 in X:
    ze=[]

    for i,x2 in enumerate(x1):
       if i<k+1: 
        ze+=torch.flatten(x2).tolist()
    coord.append(ze)

  x_train = []
  y_train = []

  x_test = []
  y_test = []
  liste=[]

  x_train, x_test, y_train, y_test= train_test_split(coord, Y, train_size=0.9)

  model = svm.LinearSVC(max_iter = 10000)
  vector=[]
  output=model.fit(x_train, y_train)

  print("score with latents {} =".format(k), model.score(x_test,y_test))
  liste.append(model.score(x_test,y_test))

  with open('graph/score_age'.format(k),"wb") as f:
    pickle.dump(liste,f)

  
from joblib import dump
dump(model,'Age_group_SVM.joblib'.format(k))
  



'''
  with open('vectors/SVM_gender_vector_{}.npy'.format(k),"wb") as f:
    pickle.dump(vector,f)

      if label["age_group"]=='0-2' or label["age_group"]=='3-6':
          Z.append(0)

      if label["age_group"]=='7-9' or label["age_group"]=='10-14':
          Z.append(1)
        
      if label["age_group"]=='15-19' or label["age_group"]=='20-29' or label["age_group"]=='30-39' or label["age_group"]=='40-49':
          Z.append(2)

      if label["age_group"]=='50-69' or label["age_group"]=='70-120':
          Z.append(3)

    for group in range(4):
    res=0
    n=0
    for i in range(len(z_test)):
      if z_test[i]==group:
        res+=(model.predict([x_test[i]])==y_test[i])
        n=n+1
    
    score=res/n
  '''

  