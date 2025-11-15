import pandas as pd
import numpy as np
import math 
import sys
from scipy.stats.mstats import mquantiles
from sklearn.model_selection import train_test_split

from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
#import torchsort
#from torchsort import soft_rank, soft_sort
import copy
import os 
from tqdm.autonotebook import tqdm
#from conformal_learning.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from conformal_learning.vgg import vgg16
from conformal_learning.densenet import densenet

import pickle 
import pdb
from conformal_learning.utils import *
from conformal_learning.help import *

from conformal_learning.losses import LDAMLoss, FocalLoss
from torch.utils.data.distributed import DistributedSampler
from conformal_learning.resnet import resnet
from conformal_learning.vgg import vgg16, vgg19_bn
from conformal_learning.densenet import densenet

from conformal_learning.sorting_nets import comm_pattern_batcher
from conformal_learning.variational_sorting_net import VariationalSortingNet
from conformal_learning.help import get_sos
from conformal_learning.smooth_conformal_prediction import smooth_aps_score, smooth_aps_score_all

from train.cifar100ManualData import load_cifar100
from train.TinyImageNetManualData import load_tiny_imagenet
from train.Cal101ManualData import load_caltech101
from train.FOOD101ManualData import load_food101
from train.iNaturalistManualData import load_inaturalist

import jax
import gc
import os
os.environ['JAX_NUMPY_DTYPE_PROMOTION'] = 'relaxed'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.3"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# rng = jax.random.PRNGKey(42) 


def save_plot(ags, scores_return, path, param_taus, data_taus):
  sns.set_style("darkgrid")
  plt.plot(torch.sort(scores_return.detach().cpu()).values, linewidth = 3, color = 'r')
  #print(param_taus)
  #print(data_taus)
  plt.hlines(y = param_taus[-1], xmin = 0, xmax = len(scores_return), colors = 'lime', label = 'From training')
  plt.hlines(y = data_taus[-1], xmin = 0, xmax = len(scores_return), colors = 'blue', label = 'From data')
  plt.xlabel('Number of data points', fontsize = 25)
  plt.ylabel('{} Scores'.format(ags.train_CP_score), fontsize = 25)
  plt.savefig(path, dpi = 100, bbox_inches='tight', pad_inches=.1)

  plt.show()
  plt.close('all')   
  
  
# def loss_cal(y, y_pred, alpha = 0.1):
#   device = y.device
#   l1 = (1 - alpha) * (y - y_pred.to(device))
#   l2 = alpha * (y_pred.to(device) - y)
#   l1 = torch.relu(l1)
#   l2 = torch.relu(l2)
#   loss = torch.mean(l1 + l2)
#   return loss

def loss_cal(y, y_pred, alpha=0.1):
  # Ensure y and y_pred are on the same device
  device = y.device
  y_pred = y_pred.to(device)
    
  # Compute the loss
  loss = torch.where(
    y_pred >= y,
    (1 - alpha) * (y_pred - y),  # Case S >= q
    alpha * (y - y_pred)         # Case S < q
  )
    
  # Return the mean of the loss
  return torch.mean(loss)


class PinballMarginal(torch.nn.Module):
  def __init__(self, ):
    super().__init__()
    self.y_pred = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
  
  def forward(self, y, tau = False):
    if tau:
      return self.y_pred
    else:
      return loss_cal(y, self.y_pred)
  
class PinballClass(torch.nn.Module):
  def __init__(self, num_classes = 100):
    super().__init__()
    self.y_pred = torch.nn.Parameter(torch.rand(num_classes), requires_grad=True)
  
  def forward(self, S, Y):
    device = Y.device
    q = self.y_pred.to(device)[Y]
    return loss_cal(S, q)
  
  
  

class ConfDataset(Dataset):
    def __init__(self, dataset_XY, Z):
        self.dataset_XY = dataset_XY
        self.Z = Z
    
    def __getitem__(self, index):
        X, Y = self.dataset_XY[index]
        Z = self.Z[index]
        return X, Y, Z
      
    def __len__(self):
        return self.Z.shape[0]


def find_scores_APS(train_proba, y_train_batch, device):
  return compute_scores_diff(train_proba, y_train_batch, device = device)

def find_scores_RAPS(train_proba, y_train_batch, device):
  return compute_scores_diff_RAPS(train_proba, y_train_batch, device = device)

def find_scores_HPS(train_proba, y_train_batch, device):
  return compute_HPS_scores(train_proba, y_train_batch, device = device)

  
def Scores_APS_all_diff(probabilities:torch.tensor)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)
  n, K = proba_values.shape

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]

  # Sorting and ranking
  ranks_array_t = soft_rank(-proba_values, device = device, regularization_strength=REG_STRENGTH)-1
  prob_sort_t = -soft_sort(-proba_values, device = device, regularization_strength=REG_STRENGTH)

  ranks_array_t = ranks_array_t.to(device)
  prob_sort_t = prob_sort_t.to(device)

  # Compute the CDF
  Z_t = prob_sort_t.cumsum(dim=1)

  prob_cum_t = soft_indexing_all_labels(Z_t, ranks_array_t)
  # Compute the PMF of the observed labels
  # Compute the conformity scores
  U = torch.rand((n, K) ,dtype=float,device=device)

  APS_differentiable_score = prob_cum_t - proba_values * U
  return APS_differentiable_score 

def Scores_RAPS_all_diff(probabilities:torch.tensor, lambda_RAPS = 0.01, k_RAPS = 5)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)
  n, K = proba_values.shape

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]

  # Sorting and ranking
  ranks_array_t = soft_rank(-proba_values, device = device, regularization_strength=REG_STRENGTH)-1
  prob_sort_t = -soft_sort(-proba_values, device = device, regularization_strength=REG_STRENGTH)

  ranks_array_t = ranks_array_t.to(device)
  prob_sort_t = prob_sort_t.to(device)

  # Compute the CDF
  Z_t = prob_sort_t.cumsum(dim=1)

  prob_cum_t = soft_indexing_all_labels(Z_t, ranks_array_t)
  # Compute the PMF of the observed labels
  # Compute the conformity scores
  U = torch.rand((n, K) ,dtype=float,device=device)

  APS_differentiable_score = prob_cum_t - proba_values * U
  
  reg_term = torch.maximum(lambda_RAPS * (ranks_array_t.to(device) - k_RAPS), torch.zeros(ranks_array_t.shape).to(device))
  
  #print(reg_term.shape, ranks_array_t.shape)
  #exit(1)
  
  RAPS_differentiable_score = APS_differentiable_score +  reg_term
  
  return RAPS_differentiable_score 

def Scores_HPS_all_diff(probabilities:torch.tensor)->torch.tensor:

  #assert probabilities.requires_grad == True
  
  device = probabilities.device
  # Break possible ties at random (it helps with the soft sorting)
  proba_values = probabilities + 1e-6*torch.rand(probabilities.shape,dtype=float,device=device)

  # Normalize the probabilities again
  proba_values = proba_values / torch.sum(proba_values,1)[:,None]


  return 1 - proba_values 
  
# Conformal Loss function
class UniformMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x, device, CDF_shift = 'KS', alpha = 0.1):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH, device = device)
    i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
    x_sorted = x_sorted.to(device)

    if CDF_shift == 'KS':
      out = torch.max(torch.abs(i_seq - x_sorted))
    elif CDF_shift == 'CM':
      out = torch.mean((i_seq - x_sorted)**2)
    elif CDF_shift == 'pinball':
      diff = x_sorted - i_seq
      diff_q1 = -alpha*diff
      diff_q2 = (1-alpha)*diff
      mask = (diff_q1 > diff_q2).int()
      out = torch.mean(mask * diff_q1 + (1 - mask) * diff_q2)
    return out

class PinballLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, theta, S, alpha):
    l1 = (1 - alpha) * (S - theta)
    l2 = alpha * (theta - S)

    return torch.relu(l1) + torch.relu(l2)
  
class ClassPinballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, num_classes=100, alpha=0.1):    
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()
    self.alpha = alpha
    self.nc = num_classes
    self.g = torch.nn.Parameter(torch.randn(num_classes) * 0.1 + (1 - alpha), requires_grad=True)
    self.pinball = PinballLoss()

  def forward(self, x, y, device):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    gY = torch.sum(torch.nn.functional.one_hot(y, num_classes=self.nc).float() * self.g.to(device), dim=-1)
    out = torch.mean(self.pinball(gY, x, self.alpha))
    #print(f"gY = {gY}")

    return out, self.g
  

class MarginalPinballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, alpha=0.1):    
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()
    self.alpha = alpha
    self.g = torch.nn.Parameter(torch.tensor([1 - alpha])*0.5, requires_grad=True)
    self.pinball = PinballLoss()

  def forward(self, scores, device):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    out = torch.mean(self.pinball(self.g.to(device), scores, self.alpha))

    return out, self.g
  
class DataMarginalPinnballLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self, alpha=0.1):   

    super().__init__()
 
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    self.alpha = alpha
    self.pinball = PinballLoss()

  def forward(self, scores, q):
    """ Compute the loss
    Parameters
    ----------
    scores : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    out = torch.mean(self.pinball(q, scores, self.alpha))
    #print(f"gY = {gY}")

    return out
  
class ExponentialMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x, CDF_shift, alpha = 0.1):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)

    U = np.random.uniform(size = batch_size)
    lam = 0.1
    X_exp = -np.log(1 - (1 - np.exp(-lam)) * U) / lam
    i_seq = soft_sort(X_exp.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
    x_sorted = x_sorted.to(device)
    i_seq = i_seq.to(device)

    if CDF_shift == 'KS':
      out = torch.max(torch.abs(i_seq - x_sorted))
    elif CDF_shift == 'CM':
      out = torch.mean((i_seq - x_sorted)**2)
    elif CDF_shift == 'pinball':
      diff = x_sorted - i_seq
      diff_q1 = -alpha*diff
      diff_q2 = (1-alpha)*diff
      mask = (diff_q1 > diff_q2).int()
      out = torch.mean(mask * diff_q1 + (1 - mask) * diff_q2)
    return out

class ChiSquareMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
    
    #i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
    x_sorted = x_sorted.to(device)

    out = torch.max(torch.abs(i_seq - x_sorted))
    return out

# def load_checkpoint(ags, model, path):
#   #print(path)
#   print(f"Loading the base model in order to fine tune")
#   if ags.arc == 'densenet100' or model_name == 'resnet110':
#     checkpoint = torch.load(path)
#     state_dict = checkpoint

#     # Remove only 'fc.weight' and 'fc.bias' from the state_dict
#     state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
#     state_dict.pop('fc.bias', None)
 
#     model.load_state_dict(state_dict)
#   else:
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint)
#   return model

# def load_checkpoint(ags, path):
#   #print(path)
#   print(f"Loading the base model in order to fine tune")
#   checkpoint = torch.load(path)
#   model.load_state_dict(checkpoint)

#   return model

def load_checkpoint(model, path):
  #print(path)
  print(f"Loading the base model in order to fine tune")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint)

  # filtered_state_dict = {k: v for k, v in checkpoint.items() if "running_mean" not in k and "running_var" not in k}
  # model.load_state_dict(filtered_state_dict, strict=False)  # Allow missing keys
  return model
        
def loss_fnc(train_rule = 'None', cls_num_list = None, num_epochs = 100, loss_type = 'CE'):
  
  if train_rule == 'None':
      #train_sampler = None  
      per_cls_weights = None 
  elif train_rule == 'Resample':
      #train_sampler = ImbalancedDatasetSampler(train_dataset)
      per_cls_weights = None
  elif train_rule == 'Reweight':
      #train_sampler = None
      beta = 0.9999
      effective_num = 1.0 - np.power(beta, cls_num_list)
      per_cls_weights = (1.0 - beta) / np.array(effective_num)
      per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
      per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)#.cuda(args.gpu)
  elif train_rule == 'DRW':
      #train_sampler = None
      idx = num_epochs // 1600
      betas = [0, 0.9999]
      effective_num = 1.0 - np.power(betas[idx], cls_num_list)
      per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
      per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
      per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)#.cuda(args.gpu)
  else:
      raise('Sample rule is not listed')

  if loss_type == 'CE':
      criterion_pred = nn.CrossEntropyLoss(weight=per_cls_weights) #.to(device)
  elif loss_type == 'IWCE':
      if ags is None:
        raise ValueError("When using IWCE loss, the `ags` argument must be provided.")
      device = torch.device(f"cuda:{ags.local_rank}" if torch.cuda.is_available() else "cpu")
      # print(f"[DEBUG] Initializing IWCELoss with reduction='mean', device={device}, CP_score={ags.train_CP_score}, T={ags.train_T}")
      criterion_pred = IWCELoss(weight=per_cls_weights, ags=ags, device=device) 
  elif loss_type == 'LDAM':
      criterion_pred = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights) #.to(device)
  elif loss_type == 'Focal':
      criterion_pred = FocalLoss(weight=per_cls_weights, gamma=1) #.to(device)
  else:
      raise('Loss type is not listed')
    
  return criterion_pred
          
def load_state(model, path = None):
  path = path
  #try:
  #  checkpoint = torch.load(path)
  #except FileNotFoundError:
  #  path = input('Please give the correct path to load the model.')
  #  checkpoint = torch.load(path)
  #except IsADirectoryError:
  #  path = input('Please give the correct path to load the model.')
  
  checkpoint = torch.load(path)    
  state_dict = checkpoint

  # # Remove only 'fc.weight' and 'fc.bias' from the state_dict
  # state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
  # state_dict.pop('fc.bias', None)
 
  model.load_state_dict(state_dict)
  # model.load_state_dict(checkpoint)
  return model
          
def load_train_objs(ags, model_name = 'resnet20', method_name = 'Baseloss', lr = 0.1, optimizer_name = 'SGD'):
  
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=False).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=False).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=False).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  # Use pre-built torchvision model
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes)  # Adjust the fully connected layer
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
    
  wd = ags.finetune_weight_decay if ags.finetune else ags.base_weight_decay
  momentum = ags.finetune_momentum if ags.finetune else ags.base_momentum

  if method_name == 'pinball-class' or method_name == 'pinball-marginal' or method_name == 'pinball_marginal_test' or ags.method == 'pinball_marginal_test_2' or method_name == 'pinball_marginal_with_Inefficiency' or method_name == 'pinball_class_with_Inefficiency':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'IW_test' or method_name == 'IW_Inefficiency' or method_name == 'only_Inefficiency' or method_name == 'only_test' or method_name == 'CPL' or method_name == 'CPL_test':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'Baseloss' or method_name == 'IWCE_Loss':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
  
  elif method_name == 'Conformal' or method_name == 'only_CDF_gap':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = wd)
      
  elif method_name == 'only_pinball_marginal':
    if optimizer_name == 'Adam':
      optimizer = optim.Adam(list(model.parameters()) + list(pinball_model.parameters()), lr=lr)
    elif optimizer_name == 'SGD':
      optimizer = optim.SGD(list(model.parameters()) + list(pinball_model.parameters()), lr=lr, momentum=momentum, weight_decay = wd)            
            
  return model, optimizer      

# def test_model(ags, model_name = 'vgg', path = None, previous = False):
#   if model_name == 'resnet110':
#     model = resnet(depth = 110, num_classes=ags.num_classes).cuda()
#   elif model_name == 'resnet20':
#     model = resnet(depth = 20, num_classes=ags.num_classes).cuda()
    
#   elif model_name == 'vgg':
#     model = vgg16().cuda()
#   elif model_name == 'vgg19_bn':
#     model = vgg19_bn().cuda()
#   elif model_name == 'densenet100':
#     model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2).cuda()
#   elif model_name == 'densenet161':
#     model = models.densenet161(pretrained=False)  
#     model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
#     model = model.cuda()
#   elif model_name == 'resnet18':
#     model = models.resnet18(pretrained=False)  
#     model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
#     model = model.cuda()
#   else:
#     raise("Please specify a correct model")
  
#   if model_name == 'densenet100' or model_name == 'resnet110':
#     checkpoint = torch.load(path)
#     state_dict = checkpoint

#     # Remove only 'fc.weight' and 'fc.bias' from the state_dict
#     state_dict.pop('fc.weight', None)  # None ensures no error if key is absent
#     state_dict.pop('fc.bias', None)
 
#     model.load_state_dict(state_dict)
#     # model.load_state_dict(checkpoint)
    
#   else:
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint)
    
#   return model 

def test_model(ags, model_name = 'vgg', path = None, previous = False, use_fc_single = False):
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=use_fc_single).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint)
    
  return model 


def test_model_epoch(ags, model_name = 'vgg', path = None, previous = False, use_fc_single = False):
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=use_fc_single).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['MODEL_STATE'])
    
  return model 

def prepare_dataloader(train_sample_dataset:Dataset, hout_sample_dataset:Dataset, batch_size:int):

  train_loader = torch.utils.data.DataLoader(
    train_sample_dataset, 
    batch_size=batch_size, 
    shuffle=False, drop_last=True, 
    sampler = DistributedSampler(train_sample_dataset))
  
  hout_loader = torch.utils.data.DataLoader(hout_sample_dataset, 
                batch_size=batch_size, shuffle=False, 
                sampler = DistributedSampler(hout_sample_dataset))


  return train_loader, hout_loader

def create_folder(ags, value):
  
  
  if ags.method == 'Baseloss':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/\
      gamma={}/weight_decay={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay)
  
  # elif ags.method == 'IWCE_Loss':
  #   file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
  #     /mu_s={}/mu_c={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/".format(
  #             ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay,\
  #             ags.finetune, ags.mu_size, ags.mu_class, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs, ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule,\
  #             ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T)    

  elif ags.method == 'IWCE_Loss':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/IWCE_rank_exponent={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay,\
              ags.finetune, ags.mu_size, ags.mu_class, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs, ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule,\
              ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T,ags.IWCE_rank_exponent)    


  elif ags.method == 'pinball-class' or ags.method == 'pinball-marginal' or ags.method == 'pinball_marginal_with_Inefficiency' or ags.method == 'pinball_class_with_Inefficiency':
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/lr_qr={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.lr_qr, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'pinball_marginal_test':
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_qr={}/lr_qr={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.mu_qr, ags.lr_qr, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'pinball_marginal_test_2' :
    
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_qr={}/lr_qr={}/qr_decay_a={}/qr_decay_b={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu_size, ags.mu_class, ags.mu_qr, ags.lr_qr, ags.qr_decay_a, ags.qr_decay_b, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    


  elif ags.method == 'Conformal':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu={}/mu_s={}/mu_p={}/train_CP_score={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu, 0.0, 0.0, ags.train_CP_score)    

  elif ags.method == 'only_CDF_gap':
      file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu={}/mu_s={}/mu_p={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.train_rho, ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.mu, 0.0, 0.0, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T)    
    

  elif ags.method == 'IW_test' or ags.method == 'IW_Inefficiency' or ags.method == 'only_Inefficiency' or ags.method == 'only_test':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay,\
              ags.finetune, ags.mu_size, ags.mu_class, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs, ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule,\
              ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T)    


  elif ags.method == 'only_pinball_marginal':
    file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/".format(
              ags.data, ags.arc, ags.baseloss, ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, ags.finetune, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T)    
  
  elif ags.method == 'CPL':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/lr_h={}/lr_lamda={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, \
              ags.finetune, ags.mu_size, ags.mu_class, ags.lr_h, ags.lr_lamda, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    

  elif ags.method == 'CPL_test':
    file_name = "/Data={}/model={}/baseloss={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/gamma={}/weight_decay={}/finetune={}\
      /mu_s={}/mu_c={}/mu_lamda={}/lr_h={}/train_CP_score={}/finetune_batchsize={}/finetune_epochs={}/finetune_lr={}/finetune_optim={}/finetune_momentum={}/finetune_lr_schedule={}/finetune_gamma={}/finetune_weight_decay={}/train_T={}/sigmoid_T={}/no_pinball_gradient={}/"\
        .format(ags.data, ags.arc, ags.baseloss, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping, ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay, \
              ags.finetune, ags.mu_size, ags.mu_class, ags.mu_lambda, ags.lr_h, ags.train_CP_score, ags.finetune_batch_size, ags.finetune_epochs,\
                ags.finetune_lr, ags.finetune_optimizer, ags.finetune_momentum, ags.finetune_lr_schedule, ags.finetune_gamma, ags.finetune_weight_decay, ags.train_T, ags.sigmid_T, True)    



  file_final = './ALLMODELS/AllModels_1/{}/final'.format(value)+file_name
  log_file = './ALLMODELS/AllModels_1/{}/logfile'.format(value)+file_name

  # file_name = file_name.replace(' ', '_')  # Replace problematic characters
  # file_final = os.path.abspath(f'./ALLMODELS/AllModels_1/{value}/final{file_name}')
  # log_file = os.path.abspath(f'./ALLMODELS/AllModels_1/{value}/logfile{file_name}')

  
  if not os.path.exists(file_final):
    
    try:
      os.makedirs(file_final)
    except FileExistsError:
      print('Already exist the main file\n')

  if not os.path.exists(log_file):
    
    try:
      os.makedirs(log_file)
    except FileExistsError:
      print('Already exist the log file\n')

  return file_final, log_file, file_name



def base_path_for_finetune(ags):
  
  file_name = "/Data={}/model={}/baseloss={}/train_rho={}/train_rule={}/batchsize={}/num_epochs={}/lr={}/lr_schedule={}/optim={}/early_stopping={}/seed={}/ntr_samples={}/base_momentum={}/\
      gamma={}/weight_decay={}/".format(
              ags.data, ags.arc, 'CE', ags.train_rho,ags.train_rule, ags.batch_size, ags.num_epochs, ags.base_lr, ags.base_lr_schedule, ags.base_optimizer, ags.early_stopping,\
              ags.seed, ags.n_tr_samples, ags.base_momentum, ags.base_gamma, ags.base_weight_decay)
      
  file_final = './ALLMODELS/AllModels_1/(Baseloss-check)/final'+file_name  

  return file_final

def create_final_data(ags, num_classes = 101):

  if ags.data == 'cifar100':
    _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_cifar100(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,num_classes=num_classes)
  
  elif ags.data == 'caltech':
     _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_caltech101(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,path='./data/caltech101',num_classes=num_classes)
  
  elif ags.data == 'inaturalist':
     _,_,_,num_train_samples,num_val_samples, train_dataset, val_dataset,_ = \
      load_inaturalist(save_path = None, n_tr = ags.n_tr_samples, n_val = ags.n_ho_samples, n_cal = ags.n_cal_samples, n_test = ags.n_test_samples,train_rho=ags.train_rho,val_rho=ags.val_rho,path='./data/inaturalist', num_classes=num_classes)
    
  if ags.method == 'Conformal':
    value = str('({}-check-1/5)'.format(ags.method))

  else:
    value = str('({}-check)'.format(ags.method))

    
  return train_dataset, val_dataset, value

def check_path(ags, file_final):
  
  if not ags.method == 'Baseloss':
    
    print(file_final + '/final_epoch={}.pt'.format(ags.finetune_epochs))
    if not os.path.exists(file_final + '/final_epoch={}.pt'.format(ags.finetune_epochs)) and os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Base model exists but fine tuned model does not exist.')
      return True
    elif os.path.exists(file_final + '/final_epoch={}.pt'.format(ags.finetune_epochs)) and os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Both finetune and basemodel exists')
      return False



  else:
    if not os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)): #'check for the base model path'
      print('Base model does not exist and need to train.')
      return True
    elif os.path.exists(base_path_for_finetune(ags)+'/final_epoch={}.pt'.format(ags.num_epochs)):
      print('Base model exists and does not need to train.')
      return False    


# @torch.no_grad()
# def Estimate_quantile_n(model, dataloader, num_classes, device, alpha, score_name):
#   _, logits, labels = get_logits_targets(model, dataloader, num_classes, device)

#   n = len(logits)
#   #print(f'n = {n}')
#   ind = int(np.ceil((1 - alpha)*(1 + n)))
  
#   soft_proba = torch.nn.Softmax(dim = 1)
#   softmax_scores = soft_proba(logits)
  
#   if score_name == 'APS':

#     sos = get_sos(num_classes)

#     scores = smooth_aps_score(softmax_scores, labels, sos=sos, device =device, dispersion=0.1, rng=None)

#     sort_scores = torch.sort(scores).values
  
#     return sort_scores[ind]    
    
#   elif score_name == 'HPS':
#     scores_all = 1 - softmax_scores
#     scores_true = scores[np.arange(n), labels]
#     sort_scores = torch.sort(scores).values
#     return sort_scores[ind]  

#   elif score_name == 'RAPS':
#     scores_all = get_RAPS_scores_all(softmax_scores)
#     scores_true = scores[np.arange(n), labels]
#     sort_scores = torch.sort(scores).values
#     return sort_scores[ind]   
  
  
#   else:
#     raise('score error')

@torch.no_grad()
def Estimate_quantile_n(model, dataloader, num_classes, device, alpha, score_name):
  _, logits, labels = get_logits_targets(model, dataloader, num_classes, device)

  n = len(logits)
  #print(f'n = {n}')
  ind = int(np.ceil((1 - alpha)*(1 + n)))
  
  soft_proba = torch.nn.Softmax(dim = 1)
  softmax_scores = soft_proba(logits)
  
  if score_name == 'APS':

    sos = get_sos(num_classes)

    scores = smooth_aps_score(softmax_scores, labels, sos=sos, device =device, dispersion=0.1, rng=None)

    sort_scores = torch.sort(scores).values
  
    return sort_scores[ind]    
    
  elif score_name == 'HPS':
    scores = 1 - softmax_scores
    scores = scores[np.arange(n), labels]
    sort_scores = torch.sort(scores).values
    return sort_scores[ind]     

  elif score_name == 'RAPS':
    scores = get_RAPS_scores_all(softmax_scores)
    scores = scores[np.arange(n), labels]
    sort_scores = torch.sort(scores).values
    return sort_scores[ind]   
  
  
  else:
    raise('score error')    

def create_optimizers(model, optimizer, classification_lr=1e-3, scale_lr=1e-3):
  """
  Create optimizers for classification and scale prediction heads.
    
  Args:
    model: The backbone model (DenseNet, ResNet, etc.) with `fc` and `fc_single` layers.
    classification_lr: Learning rate for the classification head and shared layers.
    scale_lr: Learning rate for the scale prediction head and shared layers.
    
  Returns:
    optimizer_fc: Optimizer for the classification head.
    optimizer_fc_single: Optimizer for the scale prediction head.
  """
  # Get shared parameters
  shared_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc" not in name and "fc_single" not in name  # Exclude heads
  ]
    
  # Get classification head parameters (fc)
  classification_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc" in name and "fc_single" not in name
  ]
    
  # Get scale prediction head parameters (fc_single)
  scale_params = [
    {"params": param}
    for name, param in model.named_parameters()
    if "fc_single" in name
    ]

  # Define optimizers
  optimizer_fc = type(optimizer)(shared_params + classification_params, lr=classification_lr)
  optimizer_fc_single = type(optimizer)(shared_params + scale_params, lr=scale_lr)

  return optimizer_fc, optimizer_fc_single
  
def test_model_epoch_with_q(ags, epoch, model_name = 'vgg', path = None, previous = False, use_fc_single = False):
  if model_name == 'resnet110':
    model = resnet(depth = 110, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
  elif model_name == 'resnet20':
    model = resnet(depth = 20, num_classes=ags.num_classes, use_fc_single=use_fc_single).cuda()
    
  elif model_name == 'vgg':
    model = vgg16().cuda()
  elif model_name == 'vgg19_bn':
    model = vgg19_bn().cuda()
  elif model_name == 'densenet100':
    model = densenet(depth=100, dropRate=0, num_classes=ags.num_classes, growthRate=12, compressionRate=2, use_fc_single=use_fc_single).cuda()
  elif model_name == 'densenet161':
    model = models.densenet161(pretrained=False)  
    model.classifier = torch.nn.Linear(model.classifier.in_features, ags.num_classes)  
    model = model.cuda()
  elif model_name == 'resnet18':
    model = models.resnet18(pretrained=False)  
    model.fc = torch.nn.Linear(model.fc.in_features, ags.num_classes) 
    model = model.cuda()
  else:
    raise("Please specify a correct model")
  
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['MODEL_STATE'])

  qb = np.array([item.item() if isinstance(item, torch.Tensor) else item for item in checkpoint['eval_stats']['Q_b_train']])

  # qb_splitted = np.array_split(qb, epoch)
  # qb_means = np.array([np.mean(qb_range) for qb_range in qb_splitted])


  n_q_per_epoch = len(qb) // epoch

  start_q_idx = (epoch - 1) * n_q_per_epoch
  end_q_idx = len(qb)  

  last_epoch_q = qb[start_q_idx:end_q_idx]
  q_means = np.mean(last_epoch_q)

  if ags.method == 'only_Inefficiency':

    # return model, last_epoch_q
    return model, qb[-1]    
  else:

    return model, qb[-1]
    # return model, qb_means[-1] 

# def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb, num_classes, device):

#   train_loss = []
#   test_loss = []
  
#   for q_b in qb:
#     train_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), q_b, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 
#     test_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), q_b, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 

#     train_loss.append(train_soft_size_loss.item())
#     test_loss.append(test_soft_size_loss.item())

#   train_loss = np.array(train_loss)
#   test_loss = np.array(test_loss)
#   gen_error = np.abs(train_loss-test_loss) 

#   result_df = pd.DataFrame({'General error': [gen_error]})

#   return result_df

# def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb_tr, qb_te, num_classes, device):

#   train_loss = []
#   test_loss = []

#   # assert len(qb_tr) == len(qb_te), "qb_tr and qb_te must have the same length"
  
#   # for q_tr, q_te in zip(qb_tr, qb_te):
#   num_iter = min(len(qb_tr), len(qb_te))

#   for i in range(num_iter):
#     q_tr = qb_tr[i]
#     q_te = qb_te[i]

#     train_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), q_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 
#     test_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), q_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 

#     train_loss.append(train_soft_size_loss.item())
#     test_loss.append(test_soft_size_loss.item())

#   train_loss = np.array(train_loss)
#   test_loss = np.array(test_loss)
#   gen_error = np.abs(train_loss-test_loss) 

#   result_df = pd.DataFrame({'General error': [gen_error]})

#   return result_df

# def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb_tr, qb_te, num_classes, device):

#   train_loss = []
#   test_loss = []

#   for q_tr in qb_tr:
#     # q_tr = q_tr.item() if isinstance(q_tr, torch.Tensor) else q_tr
#     train_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), q_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 
#     train_loss.append(train_soft_size_loss.item())
    
#   for q_te in qb_te:
#     # q_te = q_te.item() if isinstance(q_te, torch.Tensor) else q_te
#     test_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), q_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 

#     test_loss.append(test_soft_size_loss.item())
  
#   train_loss = np.array(train_loss)
#   test_loss = np.array(test_loss)

#   gen_error = np.abs(np.mean(train_loss)-np.mean(test_loss)) 

#   result_df = pd.DataFrame({'General error': [gen_error]})

#   return result_df

# def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb_tr, qn_te, num_classes, device):

#   train_loss = []
#   test_loss = []

#   for q_tr in qb_tr:
#     # q_tr = q_tr.item() if isinstance(q_tr, torch.Tensor) else q_tr
#     train_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), q_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 
#     train_loss.append(train_soft_size_loss.item())
    
#   star_te_loss, _, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

#   train_loss = np.array(train_loss)
#   # test_loss = np.array(test_loss)

#   gen_error = np.max(np.abs(train_loss-star_te_loss.item() ) )

#   result_df = pd.DataFrame({'General error': [gen_error]})

#   return result_df


# def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb_tr, qn_te, num_classes, device):

  
#   train_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), qb_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 

#   star_te_loss, _, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

#   # test_loss = np.array(test_loss)

#   gen_error = np.abs(train_loss.item() - star_te_loss.item()) 

#   result_df = pd.DataFrame({'Train loss': [train_loss.item()],
#                             'Test loss': [star_te_loss.item()],
#                             'General error': [gen_error]
#                            })

#   return result_df

def evaluate_sa_loss(ags, train_softmax_scores, train_targets, test_softmax_scores, test_targets, qb_tr, qn_te, num_classes, device):

  
  _, train_loss, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), qb_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 

  _, star_te_loss, _ = Estimate_size_loss_HPS_hard(test_softmax_scores.to(device), test_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

  # test_loss = np.array(test_loss)

  gen_error = np.abs(train_loss.item() - star_te_loss.item()) 

  train_softmax_scores_np = np.array(train_softmax_scores)
  test_softmax_scores_np = np.array(test_softmax_scores)
  train_targets_np = np.array(train_targets)
  test_targets_np = np.array(test_targets)
  qb_np = qb_tr.item()

  tr_cover = eval_coverage(train_softmax_scores_np, qb_np, train_targets_np)
  te_cover = eval_coverage(test_softmax_scores_np, qb_np, test_targets_np)

  result_df = pd.DataFrame({'Train loss': [train_loss.item()],
                            'Test loss': [star_te_loss.item()],
                            'General error': [gen_error],
                            'Train coverage': [tr_cover],
                            'Test coverage': [te_cover]
                           })

  return result_df

def evaluate_conftr_gap_loss(ags, train_softmax_scores, train_targets, qb_tr, num_classes, device):

  train_soft_size_loss, train_hard_size_loss, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), qb_tr, device, num_classes=num_classes, T=ags.sigmid_T, K=1.0)

  # calculate gap between soft_size_loss and hard_size_loss
  soft_gap = train_soft_size_loss.item()
  hard_gap = train_hard_size_loss.item()

  # calculate error
  gap_error = np.abs(soft_gap - hard_gap)

  # calculate coverage
  train_softmax_scores_np = np.array(train_softmax_scores)
  train_targets_np = np.array(train_targets)
  qb_np = qb_tr.item()

  tr_cover = eval_coverage(train_softmax_scores_np, qb_np, train_targets_np)

  result_df = pd.DataFrame({'Train Soft Size Loss': [train_soft_size_loss.item()],
                            'Train Hard Size Loss': [train_hard_size_loss.item()],
                            'Soft Gap': [soft_gap],
                            'Hard Gap': [hard_gap],
                            'Gap Error': [gap_error],
                            'Train Coverage': [tr_cover]
                            })

  return result_df

def evaluate_IWCE_loss(ags, tr_softmax_scores, tr_targets, te_softmax_scores, te_targets, qb, qn_tr, qn_te, num_classes, device):


  _, star_tr_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qn_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
  _, star_te_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

  _, dpsm_tr_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
  _, dpsm_te_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
       
  star_gen_error = np.abs(star_tr_loss.item() - star_te_loss.item()) 

  dpsm_gen_error = np.abs(dpsm_tr_loss.item() - star_te_loss.item())
  
  opt_gap = np.abs(dpsm_tr_loss.item() - star_tr_loss.item())

  train_softmax_scores_np = np.array(tr_softmax_scores)
  test_softmax_scores_np = np.array(te_softmax_scores)
  train_targets_np = np.array(tr_targets)
  test_targets_np = np.array(te_targets)
  qb_np = qb.item()

  tr_cover = eval_coverage(train_softmax_scores_np, qb_np, train_targets_np)
  te_cover = eval_coverage(test_softmax_scores_np, qb_np, test_targets_np)


  # Create a full DataFrame where each row is for one batch
  result_df = pd.DataFrame({
        # 'Star error': [star_gen_error],
        # 'DPSM error': [dpsm_gen_error],
        'Train loss': [dpsm_tr_loss.item()],
        'Test loss': [star_te_loss.item()],
        'opt gap': [opt_gap],
        'Train coverage': [tr_cover],
        'Test coverage': [te_cover]
        
  })

  return result_df

# def evaluate_DPSM_loss(ags, tr_softmax_scores, tr_targets, te_softmax_scores, te_targets, qb, qn_tr, qn_te, num_classes, device):


#   star_tr_loss, star_tr_hard_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qn_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
#   star_te_loss, star_te_hard_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

#   dpsm_tr_loss, dpsm_tr_hard_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
#   dpsm_te_loss, dpsm_te_hard_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
       
#   star_gen_error = np.abs(star_tr_loss.item() - star_te_loss.item()) 

#   dpsm_gen_error = np.abs(dpsm_tr_loss.item() - star_te_loss.item())
  
#   opt_gap = np.abs(dpsm_tr_loss.item() - star_tr_loss.item())

#   hard_opt_gap = np.abs(dpsm_tr_loss.item() - star_tr_loss.item())

#   # Create a full DataFrame where each row is for one batch
#   result_df = pd.DataFrame({
#         'Star error': [star_gen_error],
#         'DPSM error': [dpsm_gen_error],
#         'Train loss': [dpsm_tr_loss.item()],
#         'Test loss': [star_te_loss.item()],
#         'opt gap': [opt_gap],

#   })

#   return result_df


def evaluate_DPSM_loss(ags, tr_softmax_scores, tr_targets, te_softmax_scores, te_targets, qb, qn_tr, qn_te, num_classes, device):


  _, star_tr_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qn_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
  _, star_te_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

  _, dpsm_tr_loss, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
  _, dpsm_te_loss, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
       
  star_gen_error = np.abs(star_tr_loss.item() - star_te_loss.item()) 

  dpsm_gen_error = np.abs(dpsm_tr_loss.item() - star_te_loss.item())
  
  opt_gap = np.abs(dpsm_tr_loss.item() - star_tr_loss.item())

  train_softmax_scores_np = np.array(tr_softmax_scores)
  test_softmax_scores_np = np.array(te_softmax_scores)
  train_targets_np = np.array(tr_targets)
  test_targets_np = np.array(te_targets)
  qb_np = qb.item()

  tr_cover = eval_coverage(train_softmax_scores_np, qb_np, train_targets_np)
  te_cover = eval_coverage(test_softmax_scores_np, qb_np, test_targets_np)


  # Create a full DataFrame where each row is for one batch
  result_df = pd.DataFrame({
        # 'Star error': [star_gen_error],
        # 'DPSM error': [dpsm_gen_error],
        'Train loss': [dpsm_tr_loss.item()],
        'Test loss': [star_te_loss.item()],
        'opt gap': [opt_gap],
        'Train coverage': [tr_cover],
        'Test coverage': [te_cover]
        
  })

  return result_df

# def evaluate_DPSM_loss(ags, tr_softmax_scores, tr_targets, te_softmax_scores, te_targets, qb, qn_tr, qn_te, num_classes, device):


#   star_tr_loss, _, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qn_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
#   star_te_loss, _, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qn_te, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)   

#   dpsm_tr_loss, _, _ = Estimate_size_loss_HPS_hard(tr_softmax_scores.to(device), tr_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)          
#   dpsm_te_loss, _, _ = Estimate_size_loss_HPS_hard(te_softmax_scores.to(device), te_targets.to(device), qb, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0)    

#   for q_tr in qb:
#     # q_tr = q_tr.item() if isinstance(q_tr, torch.Tensor) else q_tr
#     train_soft_size_loss, _, _ = Estimate_size_loss_HPS_hard(train_softmax_scores.to(device), train_targets.to(device), q_tr, device, num_classes = num_classes, T = ags.sigmid_T, K = 1.0) 
#     train_loss.append(train_soft_size_loss.item())
          
       
#   star_gen_error = np.abs(star_tr_loss.item() - star_te_loss.item()) 

#   dpsm_gen_error = np.abs(dpsm_tr_loss.item() - star_te_loss.item())

#   min_dpsm_gen_error = np.min( np.abs(train_loss - star_te_loss.item()) )
  
#   opt_gap = np.abs(dpsm_tr_loss.item() - star_tr_loss.item())

#   # Create a full DataFrame where each row is for one batch
#   result_df = pd.DataFrame({
#         'Star error': [star_gen_error],
#         'DPSM error': [dpsm_gen_error],
#         'DPSM opt error': [min_dpsm_gen_error],
#         'DPSM loss': [dpsm_tr_loss.item()],
#         'Star loss': [star_te_loss.item()],
#         'opt gap': [opt_gap]
#   })

#   return result_df
    
def loss_cal_eval(y, y_pred, alpha=0.1):

  y_pred = np.array(y_pred)
  y = np.array(y)
  
  loss = np.where(
        y_pred >= y,
        (1 - alpha) * (y_pred - y),
        alpha * (y - y_pred)
    )

  return np.mean(loss)    
  
def eval_q(ags, model, data, alpha, device):
    soft_proba = torch.nn.Softmax(dim=1)
    model.eval()

    qb = []
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for source, targets in data:
            source = source.to(device)
            targets = targets.to(device)
            output = model(source)
            Temp = ags.train_T
            output /= Temp

            softmax_scores = soft_proba(output)
            scores = get_HPS_scores(softmax_scores, targets)
            tau = evaluate_quantile(scores, alpha)
            qb.append(tau.item() if isinstance(tau, torch.Tensor) else tau)

            all_scores.append(softmax_scores.detach().cpu())
            all_targets.append(targets.cpu())

    all_scores_tensor = torch.cat(all_scores, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    return all_scores_tensor, all_targets_tensor, qb


def eval_qn(ags, model, data, alpha, device):
    soft_proba = torch.nn.Softmax(dim=1)
    model.eval()

    qb = []
    all_scores = []
    all_targets = []
    hps_scores = []

    with torch.no_grad():
        for source, targets in data:
            source = source.to(device)
            targets = targets.to(device)
            output = model(source)
            Temp = ags.train_T
            output /= Temp

            softmax_scores = soft_proba(output)
            scores = get_HPS_scores(softmax_scores, targets)

            all_scores.append(softmax_scores.detach().cpu())
            hps_scores.append(scores.cpu())
            all_targets.append(targets.cpu())

    all_scores_tensor = torch.cat(all_scores, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    hps_scores_tensor = torch.cat(hps_scores, dim=0)
    # qd = Smoothquantile(hps_scores_tensor, alpha, device)
    qd = evaluate_quantile(hps_scores_tensor, alpha)

    return all_scores_tensor, all_targets_tensor, hps_scores_tensor, qd.item() if isinstance(qd, torch.Tensor) else qd


# def eval_qn(ags, model, data, alpha, device):
#     soft_proba = torch.nn.Softmax(dim=1)
#     model.eval()

#     all_scores = []  

#     with torch.no_grad():
#         for source, targets in data:
#             source = source.to(device)
#             targets = targets.to(device)
#             output = model(source)
#             Temp = ags.train_T
#             output /= Temp

#             softmax_scores = soft_proba(output)
#             scores = get_HPS_scores(softmax_scores, targets)  
#             all_scores.append(scores.cpu())

#     # Concatenate all scores and compute dataset-level quantile
#     all_scores_tensor = torch.cat(all_scores, dim=0)
#     qd = evaluate_quantile(all_scores_tensor, alpha)

#     return qd.item() if isinstance(qd, torch.Tensor) else qd

def evaluate_quantile(scores, alpha):

  n = len(scores)
  ind = int(np.ceil((1 - alpha)*(1 + n)))
  sort_scores = torch.sort(scores).values

  return sort_scores[ind]




  