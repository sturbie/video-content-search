import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

# Load trained model
MODEL_PATH = "./nlp_cv_ner_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

ID2LABEL = model.config.id2label
LABEL2ID = model.config.label2id

def extract_entities(tokens, labels):
    entities = []
    current_label = None
    current_entity = []

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_label, " ".join(current_entity)))
                current_entity = []
            current_label = label[2:]
            current_entity.append(token)
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((current_label, " ".join(current_entity)))
                current_entity = []
            current_label = None
    if current_entity:
        entities.append((current_label, " ".join(current_entity)))
    return entities

def predict(prompt):
    # Tokenize input
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    labels = [ID2LABEL[p] for p in predictions]

    # Strip special tokens
    token_label_pairs = list(zip(tokens, labels))
    filtered = [(tok, lab) for tok, lab in token_label_pairs if tok not in tokenizer.all_special_tokens]

    # Group entities
    tokens_cleaned, labels_cleaned = zip(*filtered)
    entities = extract_entities(tokens_cleaned, labels_cleaned)

    # Organize as color-class pairs
    color = None
    results = []
    for label, word in entities:
        if label == "COLOR":
            color = word
        elif label == "CLASS":
            results.append((color if color else "none", word))
            color = None  # reset color after match

    return results
