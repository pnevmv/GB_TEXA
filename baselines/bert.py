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


#///////

#Reading data folders from multipule files with this method
def load_ner_data(file_paths):
    """
    Load NER data from multiple JSON files.
    Each file contains documents with entities.
    """
    all_data= {}  #it is a dictionary which assembles all the data which are dictionary as well (data:{ "36675":{"metadata.."} , "12334":".." , ... })
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data=json.load(f)
            all_data.update(data)
            print(f'loaded {len(data)} documents from {os.path.basename(file_path)}')
        else:
            print(f"Warning: {file_path} not found")

    return all_data  


#Putting data in a format thta we want to use in NER
def prepare_documents_for_ner(data):
    """
    Convert raw data into structured format for NER.
    Each document has title and abstract as separate text segments.
    """
    document=[]

    for pmid, article in data.items():
        #Process title
        title_text= article["metadata"]["title"]
        title_entities= [e for e in article["entities"] if e["location"]=="title"]

        document.append({
            "pmid" : pmid,
            "location" : "title",
            "text" : title_text,
            "entities" : title_entities
        })

        #Process abstract
        abstract_text= article['metadata']['abstract']
        abstract_entities=[e for e in article['entities'] if e['location']== 'abstract']

        document.append({
            "pmid" : pmid,
            "location" : "abstract",
            "text" : abstract_text,
            "entities" : abstract_entities
        })

    return document   


print("✓ Data loading functions defined")    


#/////

# Load training data from three quality levels
train_files = [
    r"D:/conda_envs/Annotations/Train/gold_quality/json_format/train_gold.json",
    r"D:/conda_envs/Annotations/Train/bronze_quality/json_format/train_bronze.json",
    r"D:/conda_envs/Annotations/Train/silver_quality/json_format/train_silver.json"
]

train_data= load_ner_data(train_files)
train_documents = prepare_documents_for_ner(train_data)

print(f"\nTotal training documents: {len(train_documents)}")
print(f"Total training text segments: {len(train_data)}")


# Load dev data
dev_data = load_ner_data([r"D:/conda_envs/Annotations/Dev/json_format/dev.json"])
dev_documents = prepare_documents_for_ner(dev_data)

print(f"Total dev documents: {len(dev_documents)}")