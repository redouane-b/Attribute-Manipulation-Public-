from ast import Break
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_vdvae,load_image_tensor, normalize_ffhq_input
from pathlib import Path
from torchvision.utils import make_grid
from PIL import Image
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

#level of latent variables
k=5

#Get mean of latent variables in X and stock them in x
def get_mean(X):
  X0=[]
  X1=[]
  X2=[]
  X3=[]
  X4=[]
  for elem in X:
    X0.append(elem[0])
    X1.append(elem[1])
    X2.append(elem[2])
    X3.append(elem[3])
    X4.append(elem[4])

  x=[]
  x.append(sum(X0)/len(X0))
  x.append(sum(X1)/len(X1))
  x.append(sum(X2)/len(X2))
  x.append(sum(X3)/len(X3))
  x.append(sum(X4)/len(X4))
  return x


# load vae
vae = load_vdvae(
        conf_path="Code_python/saved_models/confs.yaml",
        conf_name="base",
        state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
        ).cuda()
vae.eval()

data=np.load("Code_python/data/data_latents_for_{}.npy".format(k),allow_pickle=True)
X_1=[]
X_2=[]
y=[]

for i,elem in enumerate(data):
    x,label=elem
    y.append(label)

    if y[i]["glasses"]=="Normal":
        X_1.append(x)
    elif y[i]["glasses"]=="None":
        X_2.append(x)

x1=get_mean(X_1)
x2=get_mean(X_2)

print(len(x1))
print(len(x2))


vector=[]
for i in range(0,k):
  vector.append((x1[i]-x2[i]).detach().cpu().numpy())

with open('vector/glasses_vector.npy',"wb") as f:
  pickle.dump(vector,f)