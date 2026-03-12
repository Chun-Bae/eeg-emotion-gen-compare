#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import random

np.random.seed(42)
random.seed(42)


# In[8]:


get_ipython().system('python -m train.train_cGAN --epochs 50')


# In[6]:


get_ipython().system('python -m eval.eval_cGAN --save-path "./experiments/c_gan_best.pth" --clf-path "./experiments/classfier_best.pth" --num-samples 100000 --batch 256')

