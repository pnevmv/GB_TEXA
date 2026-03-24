import torch
import sys, os

import json
import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Setup complete")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


#Defining the labels
ENTITY_LABELS=[
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "DDF",
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique"
]

#Creating BIO tags for entity labels
label_list=["O"]

for entity_label in ENTITY_LABELS:
    label_list.append(f'B-{entity_label}')
    label_list.append(f'I-{entity_label}')

#Defining 2 EXPRESSIONs (which generate values that is dictionary here) for getting labels and ids as tuples
label2id= {k:v for (v,k) in enumerate(label_list)}     #enumerate(label_list)=> (4,"B-animal")
id2label= {v:k for  (v,k) in enumerate(label_list)}

print(f"The 10 first label are: {label_list[:10]}")

# Model configuration
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # BioBERT for biomedical text
output_model_dir = r"D:\models\bert_biomedbert_ner"