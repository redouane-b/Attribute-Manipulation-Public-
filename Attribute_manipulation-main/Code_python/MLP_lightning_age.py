import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np



data=np.load("Code_python/data/data_latents_for_5.npy",allow_pickle=True)

X=[]
Y=[]

for i,elem in enumerate(data):
    x,label=elem
    if label["gender"]=="female":
        X.append(x)
        Y.append(0)
    elif label["gender"]=="male":
        X.append(x)
        Y.append(1)


coord=[]
for x1 in X:
  ze=[]
  for x2 in x1:
    ze+=torch.flatten(x2).tolist()
  coord.append(ze)




x_train, x_test, y_train, y_test= train_test_split(coord, Y, train_size=0.8)


dataset=[[torch.tensor(x),torch.tensor(y)] for x,y in zip(x_train,y_train) ]

def score_test(x,y):
  prediction = mlp(torch.tensor(x,dtype=torch.float32))
  acc = ((prediction.detach().numpy().round()==y).mean())
  return acc

class MLP(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(800, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
    self.ce = nn.BCELoss()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y.unsqueeze(-1).float())
    self.log('train_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
  
  
if __name__ == '__main__':
  mlp = MLP()
  trainer = pl.Trainer( gpus=torch.cuda.device_count(), deterministic=True, max_epochs=10)
  trainer.fit(mlp, DataLoader(dataset,batch_size=64,num_workers=4))




print("score =",score_test(x_train,y_train))