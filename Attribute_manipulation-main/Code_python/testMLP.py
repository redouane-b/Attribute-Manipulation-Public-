import numpy as np
import torch
from sklearn.model_selection import train_test_split
import time
import pickle
from tqdm.notebook import trange, tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning or DeprecationWarning)

#############data preprocessing###########
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

BATCH_SIZE = 64

train_iterator = data.DataLoader(x_train,y_train,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)


test_iterator = data.DataLoader(x_test,y_test,
                                batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(x_test,y_test,
                                 batch_size=BATCH_SIZE)

                                 
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
    batch_size = x.shape[0]

    x = x.view(batch_size, -1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.sigmoid(self.fc4(x))
    return x




model = Net(input_shape=len(x_test[0]))

learning_rate = 0.01
epochs = 3
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
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


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


with open('graph/loss.npy',"wb") as f:
  pickle.dump(losses,f)

with open('graph/accuracy.npy',"wb") as f:
  pickle.dump(accur,f)

with open('graph/validaton_score.npy',"wb") as f:
  pickle.dump(validation_score,f)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = 10

best_valid_loss = float('inf')

for epoch in trange(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

t2=time.time()
print("Temps=",int(t2-t1),"s")
