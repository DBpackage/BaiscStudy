# !nvidia-smi
# !pip install PyTDC
# !pip install lifelines
# !pip install rdkit-pypi

from tdc.single_pred import Tox
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import optim

# Evalutaion and visualization Functions 
import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

#### 1. Row Data Import & preprocessing ####
data = Tox(name = 'LD50_Zhu')

split = data.get_split()
split.keys()

def smiles2morgan(s, radius = 2, nBits = 1024):
    """Change SMILES data to morgan fingerprint Data
    Args:
        s (str): SMILES of a drug
        radius (int): ECFP radius
        bBits (int): size of binary representation
    Return ():
        morgan fingerprint 
    """
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits, ))
        
    return features
  
  
# Apply preprocessing function to data
for mode in ['train', 'valid', 'test']:  
  split[mode]['embedding'] = split[mode]['Drug'].apply(smiles2morgan)
#########################################



#### 2. Make class for Dataset ####
# You should make your code matched with your dataset
class ToDataset(Dataset): # Must inherite 'torch.utils.data.Dataset' 
  
  def __init__(self, df):
    self.df = df

  def __getitem__(self, index):
    vector_of_drug = self.df.iloc[index]['embedding']
    y = self.df.iloc[index]['Y']

    return vector_of_drug, y

  def __len__(self):
    return self.df.shape[0]
  
# Below datasets are not batched datasets you should make Batched Data generator with 'torch.utils.data.DataLoader'
train_dataset = ToDataset(split['train'])
valid_dataset = ToDataset(split['valid'])
test_dataset = ToDataset(split['test'])

params = {'batch_size':64,
          'shuffle':True,
          'num_workers':1,
          'drop_last':False}

# 'torch.utils.data.DataLoader'
train_generator = DataLoader(train_dataset, **params)
valid_generator = DataLoader(valid_dataset, **params)
test_generator = DataLoader(test_dataset, **params)
# Now you have three Batched Data generators 

# You should check your Batched Data generator get corrcet data with below code
# for vector_of_drug, Y in train_generator:
#   print(vector_of_drug)
#   print("vector_of_drug.shape:",vector_of_drug.shape)
#   print()
#   print(Y)
#   print(Y.shape)
#   print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now My device is :", device)
#########################################



#### 3. Making Feature Extractor Part of model ####
class MLP(nn.Sequential): # or you MUST inherite nn.Module Parents class

  def __init__(self, input_dim, output_dim, hidden_dims_list):
    super(MLP, self).__init__() # You MUST inherite the class with this line
    
    # Feature Extractor layer size:
    layer_size = len(hidden_dims_list) +1

    # Cocnat each dimension of layers
    dims = [input_dim] + hidden_dims_list + [output_dim]

    # Make your prediction MLP layer
    self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

  def forward(self, v):
    v = v.float().to(device)
    for i, layer in enumerate(self.predictor):
      v = F.relu(layer(v))
    return v
#########################################



#### 4. Making Classifier Part of model ####
class Classifier(nn.Sequential): # or you MUST inherite nn.Module Parents class

  def __init__(self, model, hidden_drug_dim, cls_hidden_dims):
    super(Classifier, self).__init__()
    # feature extractor
    self.model = model
    
    # Dropout
    self.dropout = nn.Dropout(0.1)
    
    # classifier input dimension
    self.input_dim = hidden_drug_dim
    
    # classifier hidden dimensions list
    self.hidden_dims = cls_hidden_dims
    
    # classifier layer size
    layer_size = len(cls_hidden_dims) + 1
    
    # Cocnat each dimension of layers
    dims = [self.input_dim] + self.hidden_dims + [1]
    
    # Make your Classifier FC layer
    self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

  def forward(self, V):
    Vector4predict = self.model(V)
    for i, layer in enumerate(self.predictor):
      if i == (len(self.predictor) - 1): # You MUST NOT apply Activation fncntion to the last layer
        Vector4predict = layer(Vector4predict)
      else: 
        Vector4predict = F.relu(self.dropout(layer(Vector4predict)))
    
    return Vector4predict
#########################################



#### 5. Make Your own MLP Model ####

## 5.1 Make up your model structure
input_drug_dim = 1024
hidden_drug_dim = 256
cls_hidden_dims = [1024, 1024, 512]
mlp_hidden_drug_dims = [1024, 256, 64]

MLP_FeatureExtractor = MLP(input_drug_dim, hidden_drug_dim, mlp_hidden_drug_dims)
model = Classifier(MLP_FeatureExtractor, hidden_drug_dim, cls_hidden_dims)

# print(MLP_FeatureExtractor, model)
# You can check your model layers with this code

## 5.2 set up the hyperparameters
learning_rate = 0.0001
decay         = 0.00001
train_epoch   = 15

## 5.3 Make an optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = decay) # add momentum
loss_fn = torch.nn.MSELoss()
#########################################



#### 6. Train & Validation process ####
loss_history = []
max_MSE = 10000

# 1. upload your model to device 
model = model.to(device)

# 2. initialization your model 
model_max = copy.deepcopy(model)
# Use 'copy.deepcopy' for deepcopy

# 3. Make pretty table for save your results
valid_metric_record = []
valid_metric_header = ["# epoch"]
valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
table = PrettyTable(valid_metric_header)

float2str = lambda x : '%0.4f'%x

print ("====================== Go for Training =======================") 
# 1. Training Start
time_start = time()

# 2 . For loop during epoch
for epoch in range(train_epoch):
  model.train()
  print("Current epoch : {}".format(epoch+1))
  # 2-1 . Mini-Batch Training
  for i, (Vector, label) in enumerate(train_generator):
    # 2-1-1. Upload input data to device
    Vector = Vector.float().to(device)
    
    # 2-1-2. forward pass
    Toxicity_score = model(Vector)
    n = torch.squeeze(Toxicity_score, 1)

    # 2-1-3. Loss
    loss = loss_fn(n.float(), label.float().to(device))
    loss_history.append(loss.item())

    # 2-1-4. Gradient initialization
    optimizer.zero_grad()

    # 2-1-5. Backpropagation
    loss.backward()

    # 2-1-6. Gradeint update
    optimizer.step()

  # 2-2. Mini-Batch Validation (Without Gradient tracking)
  with torch.set_grad_enabled(False): # or with torch.no_grad():
    y_pred = []
    y_label = []
    
    # set your model as evaluate model
    model.eval()

    for i, (Vector, label) in enumerate(valid_generator):
      # 2-2-1. Upload input data to device
      Vector = Vector.float().to(device)
      
      # 2-2-2. forward pass
      Toxicity_score = model(Vector)
      
      # 2-2-3. predicted value record
      logits = torch.squeeze(Toxicity_score).cpu().numpy()
      label_ids = label.cpu().numpy()
      
      # 2-2-4. Save pred and label
      y_label = y_label + label_ids.flatten().tolist()
      y_pred = y_pred + logits.flatten().tolist()

  # 2-3. Evaluation
  mse = mean_squared_error(y_label, y_pred)
  R2 = pearsonr(y_label, y_pred)[0]
  p_val = pearsonr(y_label, y_pred)[1]
  CI = concordance_index(y_label, y_pred)
  lst = ["epoch" + str(epoch)] + list(map(float2str, [mse, R2, p_val, CI]))

  # 2-4. Save results to table
  table.add_row(lst)
  valid_metric_record.append(lst)

  # 2-5. Best model update with mse score
  if mse < max_MSE:
    # 2-5-1. best model deepcopy
    model_max = copy.deepcopy(model)
    # 2-5-2. Update max MSE
    max_MSE = mse

  # 2-6. print results per epoch
  print(lst)
#########################################



#### 7. Test ####
# 1. Test 결과를 저장할 리스트 생성
y_pred = []
y_label = []

# 2. 테스트도 마찬가지로 model.eval()
model.eval()

# 3. 테스트 데이터에 대해서는 epoch돌리지 않는다.
for i, (Vector, label) in enumerate(test_generator):
  # 3.1 Vector는 device에 올린다.
  Vector = Vector.float().to(device)

  # 3.2 forward pass
  Toxicity_score = model(Vector)

  # 3.3 예측한 결과 저장
  # detach는 torch.tensor에 함께 저장되어있는 gradient graph 정보를 제거함
  pred = torch.squeeze(Toxicity_score).detach().cpu().numpy()

  # 3.4 실제 label 저장
  label_ids = label.cpu().numpy()

  # 3.5 예측값과 실제값 리스트에 저장
  y_label = y_label + label_ids.flatten().tolist()
  y_pred = y_pred + pred.flatten().tolist()

# 4. metrics 계산
mse = mean_squared_error(y_label, y_pred)
R2 = pearsonr(y_label, y_pred)[0]
p_val = pearsonr(y_label, y_pred)[1]
CI = concordance_index(y_label, y_pred)

lst = ["epoch" + str(epoch)] + list(map(float2str, [mse, R2, p_val, CI]))
#########################################



#### 8. Visualization The Results ####
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.scatter(y_label, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_label))
p2 = min(min(y_pred), min(y_label))

plt.plot([p1,p2], [p1,p2], 'b-')

plt.xlabel("True Values", fontsize=15)
plt.ylabel("Predcitions", fontsize=15)
plt.axis('equal')
plt.title("Chemical Toxicity Predcitons")
plt.show()
#########################################
