# !pip install PyTDC
# !pip install lifelines
# !pip install rdkit-pypi
# If you didn't install these packages, pls install first.

from tdc.single_pred import Tox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import copy
from prettytable import PrettyTable
from time import time

from sklearn.metircs import classification_report, roc_auc_score, roc_curve, confusion_matrix

#### 1. Data import & analysis & preprocessing ####
data = Tox(name = 'LD50_Zhu')
split = data.get_split()

# Data analysis codes :
# split.keys() : train, valid, test
# split['train'/'valid'/'test'] 
# split['train'].plot.hist()
# split['valid'].plot.hist()
# split['test'].plot.hist()

# for covert smiles seq to morgan finger print Vector(Embedding vector)
def smiles2morgan(smiles, radius=2, nBits= 1024):
  try:
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
  except:
    print("Rdkit not found this smiles for morgan : "+smiles+" Covnert to all 0 features")
    features = np.zeros((nBits, ))

  return features

for mode in ['train','valid','test']:
  split[mode]['embedding'] = split[mode]['Drug'].apply(smiles2morgan)
  split[mode]['Y_binary'] = split[mode]['Y'].apply(lambda x: int(x<2))
  # Since it's classification model, you must make new class column
  
# split['train']['embedding']
# split['train']['Y_binary']

#######################################


#### 2. Make Data generator ####
# 2.1 first you must make Dataset class
class ToDataset(Dataset): # torch.utils.data.Dataset
  def __init__(self, data):
    self.data = data
  
  def __getitem__(self, index):
    drug_vector = self.data.iloc[index]['embedding']
    y = self.data.iloc[index]['Y_binary']
  
    return drug_vector, y
  
  def __len__(self):
    return self.data.shape[0]

# 2.2 Make [train/valid/test] dataset first : theses dataset will give you Non batch_size dataset
train_dataset = ToDataset(split['train'])
valid_dataset = ToDataset(split['valid'])
test_dataset = ToDataset(split['test'])

# 2.3 set your generator's parameters then
params = {"shuffle":True,
         "batch_size":64,
         "num_workers":1,
         "drop_last":False}

# 2.4 Make [train/valid/test] data generator : these generator will give you batch_size dataset.
train_generator = DataLoader(train_dataset, **params)
valid_generator = DataLoader(valid_dataset, **params)
test_generator = DataLoader(test_dataset, **params)

# 2.5 setup your device befor make your classification model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
#######################################


#### 3. Make Classification model ####
# 3.1 Make Feature Extractor first
class MLP(nn.Module):
  
  def __init__(self, input_drug_dim, hidden_drug_dim, mlp_dims_list):
    super(MLP,self).__init__()
    layers_size = len(mlp_dims_list) + 1
    dims = [input_drug_dim] + mlp_dims_list + [hidden_drug_dim]
    self.FeatureExtractor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layers_size)])
    
  def forward(self, vector):
    vector = vector.float().to(device)
    for i, layer in enumerate(self.FeatureExtractor):
      vector = F.relu(layer(vector))
      
    return vector 
  
# 3.2 Make Classification model then
class Classifier(nn.Module):
  
  def __init__(self, feature_extractor, hidden_drug_dim, cls_dims_list):
    super(Classifier, self).__init__()
    self.feature_extractor = feature_extractor
    self.dropout = nn.Dropout(0.1)
    
    layers_size = len(cls_dims_list) +1
    dims = [hidden_drug_dim] + cls_dims_list + [2] # If your data is binary, set output layer size as 2, else fit to your class num
    self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layers_size)])
    
  def forward(self, vector):
    Vector4predict = self.feature_extractor(vector) # first, you have to get latent vector from feature extractor
    for i, layer in enumerate(self.predictor):
      if i == len(self.predictor) -1:
        Vector4predict = layer(Vector4predict)
      else:
        Vector4predict = F.relu(self.dropout(layer(Vector4predict)))
        
    return Vector4predict
  
# 3.3 set your model parameters
input_drug_dim = 1024 
hidden_drug_dim = 256
mlp_layers_list = [512, 256, 128, 64]
cls_layers_list = [128, 128, 64, 32]

# 3.4 Make a Full model
MLP_feature_extractor = MLP(input_drug_dim, hidden_drug_dim, mlp_layers_list)
model = Classifier(MLP_feature_extractor, hidden_drug_dim, cls_layers_list)

# print(model)
#######################################


#### 4. Make Optimizer & Set loss function ####
# 4.1 set your optimizer's hyper parameters
learning_rate = 0.0001
decay         = 0.00001
train_epoch   = 50

# 4.2 make optimizer & loss function
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = decay)
loss_fn = nn.CrossEntropyLoss()
#######################################


#### 5. Train & valid your model ####
# 5.1 make list for saving results
loss_history_train = []
loss_history_valid = []

# 5.2 set variable for update your best model : you can use any variable ex) accuracy, auc_roc_score, recall ...
# but here, we use accuracy score as criteria
min_acc = 0 

# 5.3 upload model to device
model = model.to(device)

# 5.4 Initialize best model
best_model = copy.deepcopy(model)

# 5.5 Make Tables 
valid_metric_record = []
valid_metric_header = ["# epoch"]
valid_metric_header.extend(['Acc','sensitivity','specificity','roc_score'])
table = PrettyTable(valid_metric_header)

float2str = lambda x : '%0.4f'%x

# 5.6 Training Start
print("____ GO FOR TRAINING ____")
t_start = time()

for epoch in range(train_epoch):
  model.train()
  
  for i, (vector, label) in enumerate(train_generator):
    vector = vector.float().to(device)
    
    output = model(vector)
    loss = loss_fn(output, label.to(device))
    loss_history_train.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  with torch.no_grad():
    y_pred = []
    y_label = []
    y_score = [] 
    
    model.eval()
    
    for i, (vector, label) in enumerate(valid_generator):
      vector = vector.float().to(device)
      output = model(vector)
      
      loss = loss_fn(output, label.to(device))
      loss_history_valid.append(loss.item())
      
      pred = output.argmax(dim=1, keepdim=True)
      score = nn.Softmax(dim = 1)(output)[:, 1]
      
      pred = pred.cpu().numpy()
      label = label.cpu().numpy()
      score = score.cpu().numpy()
      
      y_pred = y_pred + pred.flatten().tolist()
      y_label = y_lable + label.flatten().tolist()
      y_score = y_score + score.flatten().tolist()
      
# 5.7 save results
classification_metrics = classification_report(y_label, y_pred,
                                              traget_names = ["NonToxic","Toxic"],
                                              output_dict = True)
  
sensitivity = classification_metrics["Toxic"]["recall"]
specificity = classification_metrics["NonToxic"]["recall"]
accuracy = classification_metrics['accuracy']
conf_matrix = confusion_matrix(y_label, y_pred)
roc_score = roc_auc_score(y_label, y_score)
  
lst = ["epoch"+ str(epoch)] + list(map(float2str, [accuracy, sensitivity, specificity, roc_score]))
table.add_row(lst)
valid_metric_record.append(lst)

# 5.8 update your best model
if accuracy > min_acc:
  model_best = copy.deepcopy(model)
  min_acc = accuracy
#######################################


#### 6. Test ####
# test

model.eval()

y_pred = []
y_label = []
y_score = []

for i, (vector, label) in enumerate(test_generator):
    vector = vector.float().to(device)

    with torch.no_grad():

        output = model(vector)

        loss = loss_fn(output, label.to(device))

        loss_history_val.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)
        score = nn.Softmax(dim=1)(output)[:, 1]

        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        score = score.cpu().numpy()

    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + pred.flatten().tolist()
    y_score = y_score + score.flatten().tolist()

classification_metrics = classification_report(y_label, y_pred,
                                              target_names = ['NonToxic', 'Toxic'],
                                              output_dict = True)

sensitivity = classification_metrics['Toxic']['recall']
specificity = classification_metrics['NonToxic']['recall']
accuracy = classification_metrics['accuracy']
roc_score = roc_auc_score(y_label, y_score)
conf_matrix = confusion_matrix(y_label, y_pred)

print("sensitivity : {} specificity : {} accuracy : {} roc_score : {} ".format(sensitivity, specificity, accuracy, roc_score))
#######################################
