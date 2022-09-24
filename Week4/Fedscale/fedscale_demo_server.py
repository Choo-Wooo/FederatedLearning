#i!/usr/bin/env python
# coding: utf-8

# # Federated Learning for Image Classification using Fedscale

# ## Server Side

# In[1]:


import sys, os

from fedscale.core.execution.client import Client
from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.logger.execution import args

Demo_Aggregator = Aggregator(args)
### On CPU
args.use_cuda = "False"
Demo_Aggregator.run()

# In[ ]:


   # !tensorboard --logdir=./logs/demo_job --port=6007 --bind_all
