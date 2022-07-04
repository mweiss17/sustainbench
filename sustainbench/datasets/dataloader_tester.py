import sys
import os
import json
from pkgutil import get_data
import pandas as pd
from sustainbench.datasets.croptypemapping_kenya import CropTypeMappingKenyaDataset
from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import croptypemapping_kenya


import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score

ds = get_dataset(dataset='africa_crop_type_mapping', split_scheme='ghana', calculate_bands='false', version=1.0, download=True)
#ds = get_dataset(dataset='poverty', version=1.1, download=True)
#ds = get_dataset(dataset='fmow', version=1.1, download=True)
print(ds.get_subset('train'))
