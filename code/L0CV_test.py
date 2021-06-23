import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib as plt
from PIL import Image

import sys
sys.path.append("..") 
import L0CV
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 均已测试

temp = "%-15s %-15s %15s"  
print(temp % ("device", "torch version", "L0CV version"))
print(temp % (device, torch.__version__, L0CV.__version__))