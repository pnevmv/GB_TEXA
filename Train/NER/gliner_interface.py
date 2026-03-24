import json
import os
from pathlib import Path
from importlib.metadata import version

import torch
from gliner import GLiNER

print(f"GLiNER version: {version('gliner')}")

# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "numind/NuNerZero"
MODEL_NAME = "NuNerZero"

THRESHOLD = 0.6

# Modes
FINETUNE_MODEL = True
GENERATE_PREDICTIONS = False

# Data selection
USE_GOLD = True
USE_SILVER = True
USE_BRONZE = False  # turn on later if you want to experiment

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / f"{MODEL_NAME}_finetuned_T{int(THRESHOLD * 100)}"
LOG_DIR = BASE_DIR / "logs"

PATH_GOLD_TRAIN = DATA_DIR / "train_gold.json"
PATH_SILVER_TRAIN = DATA_DIR / "train_silver.json"
PATH_BRONZE_TRAIN = DATA_DIR / "train_bronze.json"
PATH_DEV = DATA_DIR / "dev.json"

PATH_ARTICLES = BASE_DIR.parent.parent / "Articles" / "json_format" / "articles_dev.json"
PATH_OUTPUT_NER_PREDICTIONS = BASE_DIR.parent.parent / "Predictions" / "NER" / "predicted_entities.json"

# Training hyperparameters
NUM_STEPS = 3000
EVAL_EVERY = 200
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

LR_ENCODER = 1e-5
LR_OTHERS = 5e-5
WARMUP_RATIO = 0.1

WEIGHT_DECAY_ENCODER = 0.0
WEIGHT_DECAY_OTHER = 0.0
MAX_GRAD_NORM = 1.0

LOSS_ALPHA = 0.75
LOSS_GAMMA = 2.0
LOSS_PROB_MARGIN = 0.0
LOSS_REDUCTION = "sum"
NEGATIVES = 1.0
MASKING = "none"

SAVE_TOTAL_LIMIT = 2
SCHEDULER_TYPE = "cosine"

# bf16 only makes sense on supported hardware; keep conservative on Mac
USE_BF16 = False

ENTITY_TYPES = [
    "anatomical location",
    "animal",
    "biomedical technique",
    "bacteria",
    "chemical",
    "dietary supplement",
    "ddf",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
]

# =========================================================
# HELPERS
# =========================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def normalize_sample(sample: dict) -> dict:
    """
    Keep the converted GLiNER-format sample intact, but normalize labels to lowercase.
    Expected input from annotations_to_gliner_format.ipynb:
      {
        "tokenized_text": [...],
        "ner": [[start_token, end_token, label], ...]
      }
    """
    ner = []
    for item in sample.get("ner", []):
        if len(item) != 3:
            raise ValueError(f"Unexpected ner item: {item}")
        s, e, label = item
        ner.append([s, e, str(label).lower()])

    return {
        "tokenized_text": sample["tokenized_text"],
        "ner": ner,
    }

def prepare_dataset(samples):
    return [normalize_sample(sample) for sample in samples]

def load_train_eval_data():
    print("## LOADING TRAINING DATA ##")

    train_parts = []

    if USE_GOLD:
        print(f"Loading gold from: {PATH_GOLD_TRAIN}")
        train_gold = load_json(PATH_GOLD_TRAIN)
        print(f"Gold samples: {len(train_gold)}")
        train_parts.extend(train_gold)

    if USE_SILVER:
        print(f"Loading silver from: {PATH_SILVER_TRAIN}")
        train_silver = load_json(PATH_SILVER_TRAIN)
        print(f"Silver samples: {len(train_silver)}")
        train_parts.extend(train_silver)

    if USE_BRONZE:
        print(f"Loading bronze from: {PATH_BRONZE_TRAIN}")
        train_bronze = load_json(PATH_BRONZE_TRAIN)
        print(f"Bronze samples: {len(train_bronze)}")
        train_parts.extend(train_bronze)

    print(f"Loading dev from: {PATH_DEV}")
    eval_data = load_json(PATH_DEV)
    print(f"Dev samples: {len(eval_data)}")

    train_data = prepare_dataset(train_parts)
    eval_data = prepare_dataset(eval_data)

    print(f"Final train samples: {len(train_data)}")
    print(f"Final eval samples: {len(eval_data)}")
    return train_data, eval_data

def build_model_for_training():
    print(f"## LOADING BASE MODEL: {MODEL_ID} ##")
    model = GLiNER.from_pretrained(MODEL_ID)
    print(f"Model type: {model.__class__.__name__}")
    return model.to(dtype=torch.float32)

def build_model_for_prediction(model_path: Path):
    print(f"## LOADING FINETUNED MODEL: {model_path} ##")
    model = GLiNER.from_pretrained(str(model_path), local_files_only=True)
    print(f"Model type: {model.__class__.__name__}")
    return model

# =========================================================
# TRAINING
# =========================================================

def finetune():
    train_data, eval_data = load_train_eval_data()
    model = build_model_for_training()

    ensure_dir(LOG_DIR)
    ensure_dir(OUTPUT_DIR.parent)

    print("## LAUNCHING TRAINING ##")

    # This follows the current GLiNER repo's train.py style.
    model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=str(OUTPUT_DIR),

        # schedule
        max_steps=NUM_STEPS,
        lr_scheduler_type=SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,

        # batches / optimizer
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=float(LR_ENCODER),
        others_lr=float(LR_OTHERS),
        weight_decay=float(WEIGHT_DECAY_ENCODER),
        others_weight_decay=float(WEIGHT_DECAY_OTHER),
        max_grad_norm=float(MAX_GRAD_NORM),

        # loss
        focal_loss_alpha=float(LOSS_ALPHA),
        focal_loss_gamma=float(LOSS_GAMMA),
        focal_loss_prob_margin=float(LOSS_PROB_MARGIN),
        loss_reduction=LOSS_REDUCTION,
        negatives=float(NEGATIVES),
        masking=MASKING,

        # logging / saving
        save_steps=EVAL_EVERY,
        logging_steps=EVAL_EVERY,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # dtype
        bf16=USE_BF16,
    )

    print(f"## TRAINING COMPLETE - MODEL SAVED TO {OUTPUT_DIR} ##")

# =========================================================
# PREDICTION
# =========================================================

def load_articles(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_entities_for_articles():
    model = build_model_for_prediction(OUTPUT_DIR)
    articles = load_articles(PATH_ARTICLES)

    ensure_dir(PATH_OUTPUT_NER_PREDICTIONS.parent)

    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES} ##")
    print(f"len(articles): {len(articles)}")

    for pmid, content in articles.items():
        title = content["title"]
        abstract = content["abstract"]

        title_entities = model.predict_entities(
            title,
            ENTITY_TYPES,
            threshold=THRESHOLD,
            flat_ner=True,
            multi_label=False,
        )
        abstract_entities = model.predict_entities(
            abstract,
            ENTITY_TYPES,
            threshold=THRESHOLD,
            flat_ner=True,
            multi_label=False,
        )

        # shift abstract offsets so they match title + " " + abstract convention
        for entity in abstract_entities:
            entity["start"] += len(title) + 1
            entity["end"] += len(title) + 1

        unique_entities = []
        seen = set()

        for entity in title_entities:
            key = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["label"],
                round(float(entity["score"]), 8),
            )
            if key not in seen:
                unique_entities.append(
                    {
                        "start_idx": entity["start"],
                        "end_idx": entity["end"],
                        "tag": "t",
                        "text_span": entity["text"],
                        "entity_label": entity["label"],
                        "score": float(entity["score"]),
                    }
                )
                seen.add(key)

        for entity in abstract_entities:
            key = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["label"],
                round(float(entity["score"]), 8),
            )
            if key not in seen:
                unique_entities.append(
                    {
                        "start_idx": entity["start"],
                        "end_idx": entity["end"],
                        "tag": "a",
                        "text_span": entity["text"],
                        "entity_label": entity["label"],
                        "score": float(entity["score"]),
                    }
                )
                seen.add(key)

        content["pred_entities"] = unique_entities

    with open(PATH_OUTPUT_NER_PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"## Predictions exported to {PATH_OUTPUT_NER_PREDICTIONS} ##")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    if FINETUNE_MODEL:
        finetune()

    if GENERATE_PREDICTIONS:
        predict_entities_for_articles()