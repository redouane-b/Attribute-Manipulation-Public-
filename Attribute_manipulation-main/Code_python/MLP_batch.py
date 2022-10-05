import numpy as np
import torch
from sklearn.model_selection import train_test_split
import time
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

########## MLP ###################
import torch
from torch import nn

class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
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

#level of latent variables
k=5

#load data
data=np.load("Code_python/testdata/data_latents_for_{}.npy".format(k),allow_pickle=True)

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



model = Net(input_shape=len(x_test[0]))

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


def train(model, iterator, optimizer, criterion, device):


  for (x, y) in zip(x_train,y_train):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    optimizer.zero_grad()
    y_pred= model(x)
    loss = criterion(y_pred, y)
    acc = score_test(y_pred, y)

    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc / len(iterator)

t2=time.time()
print("Temps=",int(t2-t1),"s")

with open('graph/loss.npy',"wb") as f:
  pickle.dump(losses,f)

with open('graph/accuracy.npy',"wb") as f:
  pickle.dump(accur,f)

with open('graph/validaton_score.npy',"wb") as f:
  pickle.dump(validation_score,f)


