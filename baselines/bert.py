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
              

#////
              


#Process training data
print("Processing training data...")
processed_train = []

for i, doc in enumerate(tqdm(train_documents, desc="Processing train")):    #tqdm is terminal processing bar and desc is the description of our bar
    processed = align_labels_with_tokens(
        doc['text'],
        doc['entities'],
        tokenizer,
        label2id
    )
    processed['pmid'] = doc['pmid']
    processed['location'] = doc['location']
    processed['text'] = doc['text']
    processed['entities'] = doc['entities']
    processed_train.append(processed)

print(f"✓ Training data processed: {len(processed_train)} documents")  


print("Processing dev data...")
processed_dev = []

for i, doc in enumerate(tqdm(dev_documents, desc="Processing dev")):
    processed= align_labels_with_tokens(
        doc['text'],
        doc['entities'],
        tokenizer,
        label2id
    )
    processed['pmid']=doc['pmid']
    processed['location'] = doc['location']
    processed['text'] = doc['text']
    processed['entities'] = doc['entities']
    processed_dev.append(processed)

print(f"✓ Dev data processed: {len(processed_dev)} documents")      


#//////


#Prepair dataset for BRET training

class NERDataset(Dataset):              #the Dataset that we imported here => from torch.utils.data import Dataset
     """Custom dataset for NER token classification."""
#in PyTorch we define a class which has these 3 important functions in order to work with Dataset 
     def __init__(self, processed_data, max_length=512):     #For getting and storing the data
         self.data= processed_data
         self.max_length= max_length

     def __len__(self):                                     #For telling how many samples we have
         return len(self.data)

     def __getitem__(self, idx):                             #For getting a specific item from data
            item = self.data[idx]
            
            # Pad or truncate to max_length
            input_ids= item['input_ids'][:self.max_length]
            attention_mask= item['attention_mask'][:self.max_length]     #for separating true tokens(1's) and paddings(0's)
            labels= item['labels'][:self.max_length]
    
            #Adding PAD if it is neccessary
            padding_length= self.max_length- len(input_ids)
            if padding_length > 0 :
                input_ids= input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask= attention_mask + [0] * padding_length
                labels= labels + [-100] * padding_length              #for padding tokens we give '-100' as label in order to be ignored by LOSS function later
    
    
            return {                                                  #PyTorch works with tensors not raw python lists so our output is a dictionary of tensors
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        
print("✓ Custom dataset class defined")    


# Create datasets
print("Creating training datasets...")

train_dataset = NERDataset(processed_train)
dev_dataset = NERDataset(processed_dev)

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Dev dataset: {len(dev_dataset)} examples")


#Setup 'data collator' for token classification -- Collator in ML means it makes some data smaples into a batch (gives padding, convert to a correct format,...)
data_collator= DataCollatorForTokenClassification(
    tokenizer= tokenizer,
    padding= True,        #it pads all of the samples untill it reaches the longest sample in that batch(a group of sample) 
    return_tensors='pt'   #return the output as PyTorch tensors
)
print("✓ Data collator initialized")



training_args = TrainingArguments(
    output_dir=output_model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,            #epoch => how many times it has seen the entire dataset. if it is high it leads to overfitting
    weight_decay=0.01,             #for preventing overfitting
    eval_strategy="epoch",         #after each epoches it evaluates the output
    save_strategy="epoch",         #after each epoches it saves the output
    load_best_model_at_end=True,   #reload the best checkpoint => a saved version of the model during training that we save in a specific time, and can be reused agin 
    push_to_hub=False,             #HuggingFace Hub is an online platform for sharing models etc.
    logging_steps=100,             #it logs after 100 steps
    save_total_limit=2,
    seed=42,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

print("✓ Training configuration ready")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Learning rate: {training_args.learning_rate}")

#////


# Initialize Trainer
print("Initializing Trainer...")
trainer= Trainer(
    model= model,
    args= training_args,
    train_dataset= train_dataset,
    eval_dataset= dev_dataset,
    processing_class= tokenizer,      #!!!! in recent models of Trainer they use "processing class" but previously it was "tokenizer"
    data_collator= data_collator
)

print("✓ Trainer initialized")

#Start training...
print("="*60)
print("Starting model training...")
print("="*60)

import time
training_start_time= time.time()

train_result= trainer.train()

training_duration= time.time()- training_start_time
print("\n" + "="*60)
print("✓ TRAINING COMPLETED!")
print("="*60)
print(f"Training time: {training_duration/60:.2f} minutes")


#Saving the new model
os.makedirs(output_model_dir, exist_ok=True)
trainer.save_model(output_model_dir)             #"Trainer" save the fine-tune model that it has trained so far in that directory
tokenizer.save_pretrained(output_model_dir)      #we also should save the used tokenizer because whenever we are using upper model as inference, we should use exactly same preprocessing- tokenizer that we used in training
                                                  #save_pretrained() saves the tokenizer to a directory in Hugging Face format to be used again with from_pretrained()
print(f"✓ Model saved to: {output_model_dir}")



#Loading the trained model for inference
inference_model= AutoModelForTokenClassification.from_pretrained(output_model_dir)
inference_tokenizer= AutoTokenizer.from_pretrained(output_model_dir)
inference_model.eval()        #switches the model from training mode to evaluation mode.

# if torch.cuda.is_available():
#   inference_model=inference_model.cuda                #later I use .cuda in predict_entities function so it would be twoso I commented it

print(f"✓ Model loaded from: {output_model_dir}")


def predict_entities(model, tokenizer, text, id2label):
    """
    Perform NER inference on a single text.
    Returns list of entities with their positions and labels.
    """
    #Tokenize
    encoding= tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
        max_length=512
    )

    offset_mapping= encoding.pop('offset_mapping')[0].numpy()
                  
    # Move to GPU if available /////  I commented it because it is overriding here using cuda
    # if torch.cuda.is_available():
    #     encoding = {k: v.cuda() for k, v in encoding.items()}

    encoding = {k: v for k, v in encoding.items()}

    #Predict without updating the weights and computing gradient
    with torch.no_grad():
      outputs= model(**encoding)    # ** opens the dictionary
      predictions= torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
                  #argmax: give me the argument with max value
                  #logits: for each token, tokenizer predict some possible classes(labels) with raw score called logit. in fact it shows which lable is probably the correct one
                  #dim=-1: do the request on the last dimension which is the labels class
                  #we return the data to cpu because numpy runs only on cpu

    # Convert predictions to labels
    predicted_labels= [id2label[pred]for pred in predictions]

    #Extract entities from BIO tags; so far we have taken the labels in BIO tags but as output we have a specific format: text, start/end char/ pure lable
    entities=[]
    current_entity= None

    for idx,( label , (start_char, end_char)) in enumerate(zip(predicted_labels, offset_mapping)):

      #Skip special tokens
      if start_char==0 and end_char==0:
        continue

      if label.startswith('B-'):
        #if there is already an entity, we should save it first. because with every 'B-' we start a new entity
        if current_entity:
          entities.append(current_entity)

        #Start a new entity
        entity_label= label[2:]  # Remove 'B-' prefix
        current_entity={
            'start_idx': start_char,
            'end_idx': end_char-1,        #in offset mapping we have [start, end). ex: PARIS -> (0,5)
            'label': entity_label,
            'text_span': text[start_char: end_char]
        }

      if label.startswith('I-') and current_entity:
        #Extend current_entity
        entity_label= label[2:]   # Remove 'I-' prefix
        if entity_label == current_entity['label']:
          current_entity['end_idx']= end_char-1
          current_entity['text_span']= text[ current_entity['start_idx'] : end_char]

      else:
        # Outside or label mismatch - save current entity
        if current_entity:
          entities.append(current_entity)
          current_entity=None


   # Save last entity if exists
    if current_entity:
        entities.append(current_entity)

    return entities

print("✓ Inference function defined")