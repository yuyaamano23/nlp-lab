import logging
from datetime import datetime
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from transformers import *
import math
import argparse
import random
import copy
import os
from nltk.tokenize import word_tokenize

from utils.nli_data_reader import NLIDataReader
from utils.logging_handler import LoggingHandler
from bert_nli import BertNLIModel
from test_trained_model import evaluate


nli_reader = NLIDataReader('datasets/AllNLI')
train_num_labels = nli_reader.get_num_labels()
msnli_data = nli_reader.get_examples('train.gz') #,max_examples=5000)
print(msnli_data)
