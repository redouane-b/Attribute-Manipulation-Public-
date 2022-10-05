import numpy as np
import torch
from sklearn.model_selection import train_test_split
import time
import pickle
import pytorch_lightning as pl

########## MLP ###################
import torch
from torch import nn
from torch.nn import functional as F


class LatentClassifier(pl.LightningModule):

  def __init__(self,input_shape):
    super(LatentClassifier,self).__init__()

    self.fc1 = nn.Linear(input_shape,80)
    self.fc2 = nn.Linear(80,40)
    self.fc3 = nn.Linear(40,10)
    self.fc4 = nn.Linear(10,1)

  def forward(self,x):


    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.sigmoid(self.fc4(x))
    return x


  def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
      return optimizer


  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('val_loss', loss)


  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('train_loss', loss)
      return loss


#level of latent variables
k=5

#load data
data=np.load("Code_python/data/data_latents_for_{}.npy".format(k),allow_pickle=True)

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



x_train = []
y_train = []

x_test = []
y_test = []

x_train, x_test, y_train, y_test= train_test_split(coord, Y, train_size=0.8)

model = LatentClassifier(input_shape=len(x_test[0]))

learning_rate = 0.01
epochs = 3


optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()


# Run the training loop
losses = []
accur = []
validation_score=[]

def score_test(x,y):
  prediction = model(torch.tensor(x,dtype=torch.float32))
  acc = (prediction.reshape(-1).detach().numpy().round() == y).mean()  
  return acc

t1=time.time()

for i in range(epochs):
  for j,(x_t,y_t) in enumerate(zip(x_train,y_train)):
  
    output = model(torch.Tensor(x_t))
    loss = loss_fn(output,torch.Tensor([y_t]))

    predicted = model(torch.tensor(x_train,dtype=torch.float32))
    acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()
    
    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if j%100==0:
      losses.append(loss)
      accur.append(acc)
      print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))
      validation_score.append(score_test(x_test,y_test))
      print("score with latents {} =".format(k),score_test(x_test,y_test))
    


t2=time.time()
print("Temps=",int(t2-t1),"s")

with open('graphe_to_del/loss.npy',"wb") as f:
  pickle.dump(losses,f)

with open('graphe_to_del/accuracy.npy',"wb") as f:
  pickle.dump(accur,f)

with open('graphe_to_del/validaton_score.npy',"wb") as f:
  pickle.dump(validation_score,f)

'''
from joblib import dump
dump(model,'SVM_gender_rbf_{}_model.joblib'.format(k))
'''

  