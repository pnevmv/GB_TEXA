# TEXA GutBrainIE 2026 System

This repository contains the TEXA submission system for [GutBrainIE 2026](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/), Task 6 of the BioASQ Lab at CLEF 2026. It extracts entities, normalized concepts, and relations from PubMed titles and abstracts about the gut-brain axis.

TEXA is a multi-stage information extraction system:

1. **Named Entity Recognition (NER)** with a fine-tuned GLiNER/NuNER Zero model.
2. **Named Entity Recognition and Disambiguation (NERD)** with a hybrid entity linker based on observed mention-URI mappings plus similarity matching over URI definitions.
3. **Relation Extraction (RE)** with a fine-tuned ATLOP document-level relation extraction model.
4. **Submission packaging** into the official JSON structures for all four GutBrainIE subtasks.

The upstream baseline is <https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline>. This repository adds the TEXA run configuration, generated predictions, evaluation files, submission package, and report material.

## At a Glance

What is already included:

- converted train/dev/test data under [`Train/`](Train/), [`Annotations/`](Annotations/), and [`Articles/`](Articles/);
- generated NER, NERD, and RE predictions under [`Predictions/`](Predictions/);
- TEXA evaluation files and the final submission archive under [`Eval/`](Eval/);
- the working-notes paper source and compiled PDF under [`report/`](report/).

What usually needs regeneration:

- model predictions, if you change thresholds, checkpoints, or input articles;
- URI definitions and similarity artifacts, if the linking resources change;
- the final submission files, if any component output changes.

Practical caveats:

- several conversion and linking steps are notebooks rather than command-line scripts;
- training and large-scale inference are GPU-oriented;
- [`Train/NER/gliner_interface.py`](Train/NER/gliner_interface.py) uses editable configuration variables;
- some generated model and embedding artifacts are large and may need external storage in a clean clone.

## Subtasks

The submitted system covers all four GutBrainIE 2026 subtasks:

| Folder | Subtask | Output |
| --- | --- | --- |
| `T611` | NER | Entity mentions with spans and labels |
| `T612` | NERD | Entity mentions with spans, labels, and normalized URIs |
| `T621` | Mention-level RE | Relations between textual entity mentions |
| `T622` | Concept-level RE | Relations between normalized concept URIs |

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`Annotations/`](Annotations/) | GutBrainIE annotations in JSON, CSV, and tabular formats. |
| [`Articles/`](Articles/) | Article title/abstract files for train, dev, and test splits. |
| [`Train/NER/`](Train/NER/) | GLiNER/NuNER Zero training and prediction code. |
| [`Train/NEL/`](Train/NEL/) | Entity-linking notebooks, URI definitions, and similarity index artifacts. |
| [`Train/RE/`](Train/RE/) | ATLOP training, inference, and relation evaluation code. |
| [`Utils/`](Utils/) | Conversion notebooks between official, GLiNER, ATLOP, and evaluation formats. |
| [`Predictions/`](Predictions/) | Intermediate and final predicted entities/relations. |
| [`Eval/`](Eval/) | Official evaluator, TEXA prediction files, and packaged submission folder. |
| [`report/`](report/) | CLEF/BioASQ working-notes paper source and compiled PDF. |

## Data

The GutBrainIE data is organized into Gold, Silver, Silver 2025, Bronze, development, and test splits. The local article files contain:

| Split | Documents |
| --- | ---: |
| Gold | 639 |
| Silver | 811 |
| Silver 2025 | 499 |
| Bronze | 2,972 |
| Development | 80 |
| Test | 80 |

The workflow uses the JSON files under [`Annotations/`](Annotations/) and [`Articles/json_format/`](Articles/json_format/). CSV and tabular versions are included for inspection and interoperability.

Entity annotations include character offsets, title/abstract location, text span, label, and URI. Relation annotations include subject and object mention information, predicate, labels, and URIs. [`Eval/evaluate.py`](Eval/evaluate.py) enforces the 13 entity labels and 17 relation labels.

## Environment

The project was developed with Python 3.10. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

For GPU runs, install a CUDA-compatible PyTorch build for your machine. The original baseline suggested CUDA 11.8 wheels:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Training and large-scale prediction are GPU-oriented. The report configuration used an NVIDIA A100 GPU.

## Quick Start

Evaluate the included development outputs after setting the four prediction path constants in [`Eval/evaluate.py`](Eval/evaluate.py), for example to `TEXA_NER_1.json`, `TEXA_NERD_1.json`, `TEXA_MENTION_LEVEL_RE_1.json`, and `TEXA_CONCEPT_LEVEL_RE_1.json`:

```bash
cd Eval
python3 evaluate.py
```

Reproduce the final submission at a high level:

1. Convert annotations with the notebooks in [`Utils/`](Utils/).
2. Generate NER predictions with [`Train/NER/gliner_interface.py`](Train/NER/gliner_interface.py).
3. Link predicted entities with [`Train/NEL/entity_linker.ipynb`](Train/NEL/entity_linker.ipynb).
4. Generate relation predictions with [`Train/RE/atlop_interface.py`](Train/RE/atlop_interface.py).
5. Merge predictions with [`Utils/merge_predictions_to_evaluation_format.ipynb`](Utils/merge_predictions_to_evaluation_format.ipynb).
6. Use the files in [`Eval/TEXA_GutBrainIE_2026/`](Eval/TEXA_GutBrainIE_2026/) as the final submission structure.

## Included Artifacts

The repository includes generated outputs as well as the code needed to recreate them.

- Intermediate predictions are under [`Predictions/`](Predictions/).
- Evaluation-ready JSON files are under [`Eval/`](Eval/).
- The final submission folder is [`Eval/TEXA_GutBrainIE_2026/`](Eval/TEXA_GutBrainIE_2026/).
- The final ZIP archive is `Eval/TEXA_GutBrainIE_2026.zip`.
- The development evaluator expects gold labels at `Annotations/Dev/json_format/dev.json`.

## Reproducing the Workflow

The full workflow is intentionally staged. Each stage writes JSON artifacts that are consumed by the next stage.

### 1. Convert Training Data

Run the conversion notebooks in [`Utils/`](Utils/) to prepare task-specific inputs:

| Notebook | Output |
| --- | --- |
| [`Utils/annotations_to_gliner_format.ipynb`](Utils/annotations_to_gliner_format.ipynb) | `Train/NER/data/*.json` |
| [`Utils/annotations_to_atlop_format.ipynb`](Utils/annotations_to_atlop_format.ipynb) | `Train/RE/data/*.json` |

For relation extraction, compose the annotated and distant training sets with:

```bash
cd Train/RE
python3 compose_training_sets.py
```

This creates:

```text
Train/RE/data/train_annotated.json
Train/RE/data/train_distant.json
```

### 2. Train or Run NER

The NER entry point is [`Train/NER/gliner_interface.py`](Train/NER/gliner_interface.py). It loads `numind/NuNerZero` and can either fine-tune the model or generate predictions.

Important configuration variables are near the top of the file:

```python
THRESHOLD = 0.6
finetune_model = True
generate_predictions = False
PATH_ARTICLES = "../../Articles/json_format/articles_dev.json"
PATH_OUTPUT_NER_PREDICTIONS = "../../Predictions/NER/predicted_entities_dev.json"
```

The report configuration fine-tuned GLiNER for 3,000 steps with batch size 8. A development-set threshold sweep selected `0.65` for the submitted run. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for details.

Run from the NER directory:

```bash
cd Train/NER
python3 gliner_interface.py
```

Generated NER predictions are stored under [`Predictions/NER/`](Predictions/NER/). Use these utility notebooks to convert them for evaluation and relation extraction:

| Notebook | Purpose |
| --- | --- |
| [`Utils/NER_predictions_to_evaluation_format.ipynb`](Utils/NER_predictions_to_evaluation_format.ipynb) | Convert GLiNER output to official NER/NERD-style entity JSON. |
| [`Utils/NER_predictions_to_atlop_format.ipynb`](Utils/NER_predictions_to_atlop_format.ipynb) | Convert predicted entities to ATLOP input format. |

### 3. Run Entity Linking

The NERD component is implemented in [`Train/NEL/entity_linker.ipynb`](Train/NEL/entity_linker.ipynb). It combines:

- exact matching from mention-URI mappings observed in the annotation collections;
- URI definitions and ontology-derived descriptions in [`Train/NEL/definitions/`](Train/NEL/definitions/);
- similarity matching with a biomedical embedding index for mentions not covered by exact matching.

If the definition files are missing or need to be regenerated, run:

```text
Train/NEL/definitions/generate_definitions.ipynb
```

The current repository includes linked predictions in:

```text
Predictions/NEL/predicted_entities_dev.json
Predictions/NEL/predicted_entities_test.json
```

### 4. Train or Run Relation Extraction

The RE entry point is [`Train/RE/atlop_interface.py`](Train/RE/atlop_interface.py). It uses a DocRED-style representation and a fine-tuned ATLOP model.

Training uses:

- Gold, Silver, and Silver 2025 as manually annotated data;
- Bronze as distant supervision.

A typical training command is:

```bash
cd Train/RE
python3 atlop_interface.py \
  --data_dir data \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file predicted_entities_dev_0.65_atlop_format.json \
  --save_path outputs \
  --num_class 18
```

To run inference from an existing checkpoint:

```bash
cd Train/RE
python3 atlop_interface.py \
  --data_dir data \
  --test_file predicted_entities_test_0.65_atlop_format.json \
  --save_path outputs \
  --load_path outputs \
  --load_checkpoint best.ckpt \
  --pred_file results_test_0.65.json \
  --num_class 18
```

The final packaged run is named `t065e20`: `t065` refers to the NER threshold `0.65`, and `e20` refers to the 20-epoch RE run used for the submitted package. In this checkout, the corresponding RE inference command is:

```bash
cd Train/RE
python3 atlop_interface.py \
  --data_dir data \
  --test_file predicted_entities_test_0.65_atlop_format.json \
  --save_path outputs \
  --load_path outputs \
  --load_checkpoint best.ckpt \
  --pred_file results_test_0.65_ta.json \
  --num_class 18
```

The repository currently includes RE artifacts in [`Train/RE/outputs/`](Train/RE/outputs/) and final relation predictions in [`Predictions/RE/`](Predictions/RE/).

### 5. Merge and Package Outputs

Use [`Utils/merge_predictions_to_evaluation_format.ipynb`](Utils/merge_predictions_to_evaluation_format.ipynb) to combine entity, linked-entity, and relation predictions into the official task files.

The TEXA submission files in this checkout include:

```text
Eval/TEXA_NER_1.json
Eval/TEXA_NERD_1.json
Eval/TEXA_MENTION_LEVEL_RE_1.json
Eval/TEXA_CONCEPT_LEVEL_RE_1.json
```

Additional run variants are present with suffixes such as `_ta`, `_e20`, and `_t065e20`. The packaged final submission is under [`Eval/TEXA_GutBrainIE_2026/`](Eval/TEXA_GutBrainIE_2026/).

## Results

On the development set, the submitted TEXA configuration reached micro-F1 scores of `0.8238` for NER, `0.6316` for NERD, `0.3966` for mention-level RE, and `0.2316` for concept-level RE. The pattern reflects the staged design: entity detection is strongest, URI linking adds another source of error, and concept-level relation extraction is the most demanding target.

Detailed development tables, including macro/micro scores and the NER threshold sweep, are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
