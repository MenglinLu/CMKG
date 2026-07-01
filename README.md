# CMKG: Compact Knowledge-Guided Model for Disease Diagnosis

## Repository Structure

| File | Description |
|------|-------------|
| `config.py` | Configuration class containing hyperparameters, file paths, and model settings |
| `model.py` | Model architectures: `ModelBert` (shared BERT encoder) and `ModelV30` (semantic mapping + classification layers) |
| `main.py` | Main training script including InfoNCE loss implementation, dataset classes, and train/validation/test epoch functions |
| `preprocess.py` | Data preprocessing pipeline: text cleaning, label encoding, and dataset construction (`BuildDataset`) |
| `evaluation.py` | Evaluation metrics computation (accuracy, precision, recall, F1-score) |
| `pytorchtools.py` | Utility classes including `EarlyStopping` for training regularization |
| `langconv.py` | Traditional-to-Simplified Chinese text conversion utilities |
| `zh_wiki.py` | Chinese Wikipedia-related helper functions |

## Key Implementation Details

- **Two-stage training**: BERT is first fine-tuned on chief complaints (Stage 1), then frozen; the mapping and classification layers are trained jointly with combined contrastive + classification loss (Stage 2)
- **Contrastive learning**: InfoNCE loss with L2-normalized embeddings; all C-1 negative knowledge texts are used per sample
