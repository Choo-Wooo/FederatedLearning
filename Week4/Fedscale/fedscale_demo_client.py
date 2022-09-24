#!/usr/bin/env python
# coding: utf-8

# # Federated Learning for Image Classification using Fedscale

# ## Client Side

# In[1]:


import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os

from fedscale.core.execution.client import Client
from fedscale.core.execution.executor import Executor
from fedscale.core.logger.execution import args
### On CPU
args.use_cuda = "False"
Demo_Executor = Executor(args)
Demo_Executor.run()
