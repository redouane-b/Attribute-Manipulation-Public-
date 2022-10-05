from ast import Break
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import load_vdvae,load_image_tensor, normalize_ffhq_input
from pathlib import Path
from torchvision.utils import make_grid
from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

#level of latent variables
k=5
src=9
step=0.3
nk=30

def translation(point,vector,constant):
  point = [p+constant*v for p,v in zip(point,vector)]
  return point

# load vae
vae = load_vdvae(
        conf_path="Code_python/saved_models/confs.yaml",
        conf_name="base",
        state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
        map_location='cpu')
vae.eval()

def get_latents(vae, x):
    with torch.no_grad():
        stats = vae.forward_get_latents(x) # [{'z' : ..., 'kl' : ...}]
    latents = [d['z'] for d in stats] # [z1, ..., zL] z1.shape = [1,C,H,W]
    return latents[0:50]


x = load_image_tensor("Code_python/images/7.png")
y = load_image_tensor("Code_python/images/7wg.png")

x_f = normalize_ffhq_input(x)
y_f = normalize_ffhq_input(y)

ZS1 = get_latents(vae, x_f)
ZS2 = get_latents(vae, y_f)

ZS1=ZS1[0:nk]
ZS2=ZS2[0:nk]

vector=[]

for e1,e2 in zip(ZS1,ZS2):
  vector.append(e1-e2)


l=[]

xp = load_image_tensor("Code_python/images/{}.png".format(src))
xp_f = normalize_ffhq_input(xp)
ZSp = get_latents(vae, xp_f)
p=ZSp

with torch.no_grad():
  for i in range(6):
    elem=vae.forward_samples_set_latents(3, p)
    l.append(elem)
    p=translation(p,vector,step)


l = torch.cat(l)
grid = make_grid(l, nrow=3)
grid=grid.permute(1, 2, 0)
grid=grid.detach().cpu().numpy()


outpath = Path("res/").joinpath('image_test_using_{}_levels_{}_from_im{}_with_step{}.png'.format(k,nk,src,step))
im = Image.fromarray(grid)
im.save(outpath)
 
