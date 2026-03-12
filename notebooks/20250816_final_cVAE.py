#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random
import time
import gc
import os
from models.classifier import Classifier
from models.cVAE import CondVAE1D

from utils.device_selection import device_selection
from utils.load_classifier import load_classifier
from data_processing.data_load import initial_dreamer_load
from data_processing.data_object import EEGCondDataset
from data_processing.data_spilt import gen_data_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# ---- scipy ----
import scipy.io

# ---- torch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader

np.random.seed(42)
random.seed(42)


# In[28]:


get_ipython().system('python -m train.train_cVAE --epochs 30')


# In[29]:


get_ipython().system('python -m eval.eval_cVAE')


# In[ ]:




