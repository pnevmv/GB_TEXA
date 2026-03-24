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


#/////


# Initialize tokenizer and model
print("Initializing BERT tokenizer and model...")

tokenizer= AutoTokenizer.from_pretrained(model_name)
model= AutoModelForTokenClassification.from_pretrained(     #where we tell the model what labels it should work with
    model_name,
    num_labels= len(label_list),
    id2label=id2label,
    label2id=label2id
)

print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"✓ Model loaded with {model.num_labels} labels")
print(f"  Vocabulary size: {tokenizer.vocab_size}")


#/////

def align_labels_with_tokens(text, entities, tokenizer, label2id):
    """
    Create BIO tags for tokenized text based on character-level entity annotations.
    Uses offset mapping to align character positions with token positions.
    """
    #Tokenize and get offset mapping
    encoding= tokenizer(              #encoding is the output if the tokenizer (a dictionary)
        text,
        add_special_tokens= True,
        return_offsets_mapping= True,
        truncation=True,              #if the number of tokens exceeds from max_length it truns them
        max_length=512     
    )

    tokens= tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offset_mapping=  encoding['offset_mapping']  

    #initial all labels as 'O' outside
    labels= ['O']* len(tokens)

    #sort entities by start position to handle overlaps (first the intial or longer entities)
    sorted_entities= sorted(entities, key=lambda e: (e['start_idx'], -(e['end_idx'] - e['start_idx'])))    #sorted method needs a key to sort the items based on it. here it sorts the entities which come first, if their start index were same, it choses the longest one (second priority)

    #track which tokens have been labeled 
    labeled_positions= set()              #we define a set instead of array because when we want to check if it is repeated or  not, since set uses hash table it is faster than array exploring. also set doesn't add repeated values

    #for each entity with labels we find the associated tokens in our text
    for entity in sorted_entities :
        entity_start= entity['start_idx']
        entity_end= entity['end_idx']
        entity_label= entity['label']

        #find tokens that overlap with this entity 
        entity_token_star= None
        entity_token_end=None

#-------------------------------------------------------------------------- This section finds out which tokens belongs to one entity. for example: New York City - we have New, York, City tokens[1-3] all belonging to one entity labeled as LOC. So entity_token_star stays at 1 and entity_token_end become 3 after 3 iteration.       
        for idx, (start_token, end_token) in enumerate(offset_mapping):
            #skip special chars
            if start_token==0 and end_token== 0:
                continue
            #check if token overlaps with entity
            if start_token< entity_end and end_token> entity_start :  #it doesn't have to be fully inside the entity, it also can only overlap -even one commen char 
                if entity_token_star is None:
                    entity_token_star= idx
                entity_token_end= idx
#------------------------------------------------------------------------- we get the labels combined with BIO tags
         #Apply BIO tagging
        if entity_token_star is not None and entity_token_end is not None:
            for index in range (entity_token_star, entity_token_end+1):
                    # Only label this token if not already labeled (handle overlaps)
                if index not in labeled_positions:
                    if index == entity_token_star :
                        labels[index]= f'B-{entity_label}'
                    else:
                        labels[index]=f'I-{entity_label}'
    
                    labeled_positions.add(index)    
#---------------------------------------------------------------------------
    # Convert labels to IDs
    label_ids = [label2id.get(label, label2id['O']) for label in labels]     #GET method gets labels but if it wasn't inside label2id dictionary that we have defined in the begining it will write 'O' as DEFAULT value
        
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': label_ids,
        'tokens': tokens
    }      
            
print("✓ BIO tag generation function defined")                  
              