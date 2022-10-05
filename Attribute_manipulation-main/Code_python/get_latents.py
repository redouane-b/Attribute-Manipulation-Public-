"""
Get the latent representation of the images and stock them in a file (latent.py)
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mmap import mmap
import numpy as np
from utils import load_vdvae, load_image_tensor, save_image_tensor, normalize_ffhq_input
from torchvision.utils import make_grid
import torch
import time
import pandas as pd


def get_latents(vae, x):
    with torch.no_grad():
        stats = vae.forward_get_latents(x)
    latents = [d['z'] for d in stats]
    return latents


#Load the tensor with the weights
vae = load_vdvae(
    conf_path="Code_python/saved_models/confs.yaml",
    conf_name="base",
    state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
    map_location=torch.device('cpu'))


#Loop over all images in dataset
data = np.load('/beegfs/rbouazza/ffhq-256.npy',mmap_mode='r+')
labels=pd.read_csv("Code_python/ffhq_aging_labels.csv")


Z1=[]
Z2=[]
Z3=[]
Z4=[]
Z5=[]
Zall=[]

t1=time.time()

for i,image in enumerate(data):
    #get the latent representation
    x= torch.tensor(image)
    x_n = normalize_ffhq_input(x)
    x_n=torch.permute(x_n, (2, 1, 0))
    x_n=torch.unsqueeze(x_n, 0)
    
    z=get_latents(vae,x_n)

    l1=[z[0:1],labels.iloc[i]]
    l2=[z[0:2],labels.iloc[i]]
    l3=[z[0:3],labels.iloc[i]]
    l4=[z[0:4],labels.iloc[i]]
    l5=[z[0:5],labels.iloc[i]]
    lall=[z[0:8],labels.iloc[i]]


    Z1.append(l1)
    Z2.append(l2)
    Z3.append(l3)
    Z4.append(l4)
    Z5.append(l5)
    Zall.append(lall)

    nZ1=np.array(Z1,dtype=object)
    nZ2=np.array(Z2,dtype=object)
    nZ3=np.array(Z3,dtype=object)
    nZ4=np.array(Z4,dtype=object)
    nZ5=np.array(Z5,dtype=object)
    nZall=np.array(Zall,dtype=object)


    np.save('data_latents_10.npy',nZall)

    if i%100==0:
        print("step {} done".format(i))

t2=time.time()
print("Temps=",int(t2-t1),"s")

