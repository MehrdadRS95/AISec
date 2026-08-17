import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Load variables from .env file (HF_TOKEN)
load_dotenv()

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lowered batch size to 4 to prevent GPU Out-Of-Memory (OOM) on laptops
batch_size = 4
learning_rate = 2e-5
num_epochs = 3

# ==========================================
# 2. LOAD AND PREPARE THE DATASET
# ==========================================
print("Loading dataset...")
dataset = load_dataset("deepset/prompt-injections")

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(examples):
    # Truncate long prompts to 512 tokens (BERT's maximum)
    clean_texts = [str(text) for text in examples["text"]]
    return tokenizer(clean_texts, padding="max_length", truncation=True, max_length=512)


print("Tokenizing data...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 1. Remove raw text column
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 2. Rename 'label' -> 'labels' so BERT automatically computes loss
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 3. Format dataset outputs as PyTorch Tensors
tokenized_datasets.set_format("torch")

# 4. Single, clean DataLoader definition
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

# ==========================================
# 3. INITIALIZE THE MODEL
# ==========================================
print("Loading pre-trained BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# ==========================================
# 4. OPTIMIZER & SCHEDULER
# ==========================================
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
print("Starting training...")
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move tensor batch to target device (CPU/GPU)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimizer & scheduler updates
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    print(f"Epoch {epoch + 1} complete.")

# ==========================================
# 6. SAVE MODEL & TOKENIZER
# ==========================================
print("Saving fine-tuned model...")
model.save_pretrained("./mcp-bert-guardrail")
tokenizer.save_pretrained("./mcp-bert-guardrail")
print("Model saved to ./mcp-bert-guardrail")

# ==========================================
# 7. EVALUATE ON THE TEST SET
# ==========================================
print("Running evaluation on the test dataset...")
model.eval()

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(batch["labels"].cpu().numpy())

# ==========================================
# 8. CALCULATE METRICS & ANALYSIS
# ==========================================
accuracy = accuracy_score(all_true_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

print("\n" + "=" * 40)
print("🛡️ AI GUARDRAIL SECURITY REPORT 🛡️")
print("=" * 40)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print("=" * 40)

print("\nDetailed Breakdown:")
print(classification_report(all_true_labels, all_predictions, target_names=["Safe (0)", "Injection (1)"]))

# ==========================================
# 9. VISUALIZE CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Predicted Safe", "Predicted Injection"],
    yticklabels=["Actually Safe", "Actually Injection"]
)
plt.title("Guardrail Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()