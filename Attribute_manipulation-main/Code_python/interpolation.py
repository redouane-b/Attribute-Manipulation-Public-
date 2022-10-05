"""
tranfer the attribute of a ref image encoded in the "level" latent variable to a source image 
"""

from utils import load_vdvae, load_image_tensor, save_image_tensor, normalize_ffhq_input
import argparse
import os
from pathlib import Path
from torchvision.utils import make_grid
import torch

def get_latents(vae, x):
    with torch.no_grad():
        stats = vae.forward_get_latents(x) # [{'z' : ..., 'kl' : ...}]
    latents = [d['z'] for d in stats] # [z1, ..., zL] z1.shape = [1,C,H,W]
    return latents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Code_python/images/7.png")
    parser.add_argument("--ref", default="Code_python/images/12.png")
    parser.add_argument("--level", type=int, default=4, help="level in the latent variable hierarchy to interpolate")
    parser.add_argument("--nstep", type=int, default=10)
    parser.add_argument("--samples_per_latent", type=int, default=10)
    parser.add_argument("--outdir", default="results/interpolation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--samples_per_step", type=int, default=6)
    args = parser.parse_args()

    # load vae
    print("loading vae \n ")
    vae = load_vdvae(
        conf_path="Code_python/saved_models/confs.yaml",
        conf_name="base",
        state_dict_path="Code_python/saved_models/ffhq256-iter-1700000-model-ema.th",
        map_location=torch.device('cpu')
        )
    vae.eval()
    print("loaded \n ")


    # load images tensor
    print('loading image \n ')
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    xs = load_image_tensor(args.source) #uint8
    xr = load_image_tensor(args.ref)#uint8
    xs_f = normalize_ffhq_input(xs)# float
    xr_f = normalize_ffhq_input(xr) # float
    print("loaded \n ")

    # get latents variable
    ZS = get_latents(vae, xs_f)
    ZR = get_latents(vae, xr_f)

    # list of images
    #l = [xs]
    l = []
    zs_l = ZS[:args.level+1]
    zr_l = ZR[:args.level+1]

    ZI=ZS

    for s in range(args.nstep+1):
        print("step=",s)
        t = s / args.nstep
        for m, (zs, zr) in enumerate(zip(zs_l, zr_l)):
            ZI[m] = (1-t) * zs + t * zr
        with torch.no_grad():
            l.append(vae.forward_samples_set_latents(1, ZI))



    l = torch.cat(l)
    grid = make_grid(l, nrow=args.nstep+1)
    sname = Path(args.source).stem
    refname = Path(args.ref).stem
    outpath = Path(args.outdir).joinpath('{}_{}_l__lmax_{}.png'.format(sname, refname, args.level,#args.level_max
    ))

    save_image_tensor(grid, str(outpath))
    print("image saved")
