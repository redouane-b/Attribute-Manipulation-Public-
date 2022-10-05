import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_vdvae,load_image_tensor, normalize_ffhq_input
from pathlib import Path
from torchvision.utils import make_grid
from PIL import Image
import random
from joblib import dump, load

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

def list_to_tensor(list1):
  e1=list1[0:16]
  e2=list1[16:32]
  e3=list1[32:288]
  e4=list1[288:544]
  e5=list1[544:800]

  m1=torch.reshape(torch.Tensor(e1),(1,16,1,1))
  m2=torch.reshape(torch.Tensor(e2),(1,16,1,1))
  m3=torch.reshape(torch.Tensor(e3),(1,16,4,4))
  m4=torch.reshape(torch.Tensor(e4),(1,16,4,4))
  m5=torch.reshape(torch.Tensor(e5),(1,16,4,4))

  tens=[m1,m2,m3,m4,m5]
  return tens

def tensor_to_list(tensor1,k):
  lis=[]
  
  e0=torch.squeeze(tensor1[0])
  lis+=(torch.reshape(e0,(16,)).tolist())
  if k>1:
    e1=torch.squeeze(tensor1[1])
    lis+=(torch.reshape(e1,(16,)).tolist())
    if k>2:
      e2=torch.squeeze(tensor1[2])
      lis+=(torch.reshape(e2,(256,)).tolist())
      if k>3:
        e3=torch.squeeze(tensor1[3])
        lis+=(torch.reshape(e3,(256,)).tolist())
        if k>4:
          e4=torch.squeeze(tensor1[4])
          lis+=(torch.reshape(e4,(256,)).tolist())

  

  return lis

def sample_latents(nbatch,latents,k,t):
  with torch.no_grad():
        next_lat=vae.forward_sample_next_latent(nbatch,latents,k,t=t)
  return next_lat  
   


#level of latent variables
k=5
temp=0.8
n_im=18

vae = load_vdvae(
        conf_path="Code_python/saved_models/confs.yaml",
        conf_name="base",
        state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
        map_location='cpu')
vae.eval()

list_lat=[]

for k in range(1,6):
  with open('vectors/SVM_gender_vector_{}.npy'.format(k),"rb") as f:
    list_lat.append(pickle.load(f))


l=[]
sampleL=[]

with torch.no_grad():
  for i in range(n_im):
    sample=[]

    for k in range(1,5):
      k=4
      print("étape i={} pour k={}".format(i,k))
      
      model = load('Code_python/models/Age_group_SVM.joblib')

      sample=sample_latents(1,sample,k,t=temp)
      sampleL=tensor_to_list(sample,k+1)
      #vect1=list_lat[k][0][0].tolist()
      print(model.predict([sampleL]))
      n=1
      #< for men; > for women sum([x*y for x,y in zip(vect1,sampleL[k])])<-list_lat[k][1][0]
      while model.predict([sampleL])!=0 and n<100:
        sample=sample[:-1]
        sample=sample_latents(1,sample,k,t=temp)
        sampleL=tensor_to_list(sample,k+1)
        n+=1
        
      print("Trouvé aprés {} essai".format(n))

      print(model.predict([sampleL]))
        
    




    sampleT=list_to_tensor(sampleL)
    elem=vae.forward_samples_set_latents(1, sampleT,t=temp)
    l.append(elem)
    

l = torch.cat(l)
grid = make_grid(l, nrow=6)
grid=grid.permute(1, 2, 0)
grid=grid.detach().cpu().numpy()


outpath = Path("res/").joinpath('SVM_age_testd_gen_using_{}_and_temp_{}_{}.png'.format(k,temp,n_im))
im = Image.fromarray(grid)
im.save(outpath)
 
