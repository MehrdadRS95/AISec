import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # FIX: Imported directly from PyTorch
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns


import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# Verify it loaded (optional check)
# print("HF Token Loaded:", bool(os.getenv("HF_TOKEN")))


# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 16
learning_rate = 2e-5
num_epochs = 3



# ==========================================
# 2. LOAD AND PREPARE THE DATASET
# ==========================================
print("Loading dataset...")
# This automatically downloads the deepset/prompt-injections dataset
dataset = load_dataset("deepset/prompt-injections")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    # Truncate long prompts to 512 tokens (BERT's maximum)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


print("Tokenizing data...")
# Apply tokenization to the whole dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove the raw text column (PyTorch only wants numbers)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# Rename 'label' to 'labels' (which is what the BERT model expects)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Set the format to PyTorch tensors
# tokenized_datasets.set_format("torch")


# 1. DELETE THIS LINE:
# tokenized_datasets.set_format("torch")

# 2. Add this custom formatting function:
def custom_collate(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch])
    }

# 3. Update your DataLoaders to use the custom function:
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=custom_collate
)

eval_dataloader = DataLoader(
    tokenized_datasets["test"],
    batch_size=batch_size,
    collate_fn=custom_collate
)


# tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



# Create DataLoaders to feed data to the model in batches
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

# ==========================================
# 3. INITIALIZE THE MODEL
# ==========================================
print("Loading pre-trained BERT model...")
# Load BERT with a 2-class classification head (0: Safe, 1: Injection)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# ==========================================
# 4. OPTIMIZER & SCHEDULER
# ==========================================
# AdamW is the standard optimizer for transformer models
optimizer = AdamW(model.parameters(), lr=learning_rate)

num_training_steps = num_epochs * len(train_dataloader)
# A learning rate scheduler helps the model settle into the optimal weights
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)



# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
print("Starting training...")
progress_bar = tqdm(range(num_training_steps))

model.train() # Put model in training mode
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move all tensors in the batch to the GPU (if available)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass: Feed the data into the model
        outputs = model(**batch)

        # Calculate loss (how wrong the model was)
        loss = outputs.loss

        # Backward pass: Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()
        lr_scheduler.step()

        # Clear gradients for the next step
        optimizer.zero_grad()
        progress_bar.update(1)

    print(f"Epoch {epoch + 1} complete.")

# ==========================================
# 6. SAVE YOUR SECURITY GUARDRAIL
# ==========================================
print("Saving fine-tuned model...")
# Save the model and tokenizer to a local directory
model.save_pretrained("./mcp-bert-guardrail")
tokenizer.save_pretrained("./mcp-bert-guardrail")
print("Done! Your model is ready to protect your MCP server.")


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 7. EVALUATE ON THE TEST SET
# ==========================================
print("Running evaluation on the test dataset...")

# Put the model in evaluation mode (turns off training-specific features like dropout)
model.eval()

# Lists to store the model's guesses and the actual correct answers
all_predictions = []
all_true_labels = []

# We use torch.no_grad() because we don't need to calculate gradients for testing.
# This makes it run much faster and uses less memory.
with torch.no_grad():
    for batch in eval_dataloader:
        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # Get the model's raw output scores (logits)
        outputs = model(**batch)
        logits = outputs.logits

        # The model outputs two scores (Safe vs Malicious).
        # torch.argmax picks the index of the highest score (0 or 1)
        predictions = torch.argmax(logits, dim=-1)

        # Move the results back to the CPU and add them to our lists
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(batch["labels"].cpu().numpy())

# ==========================================
# 8. CALCULATE METRICS & ANALYSIS
# ==========================================
# Calculate core metrics using scikit-learn
accuracy = accuracy_score(all_true_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='binary')

print("\n" + "="*40)
print("🛡️ AI GUARDRAIL SECURITY REPORT 🛡️")
print("="*40)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print("="*40)

# Generate a detailed classification report
print("\nDetailed Breakdown:")
print(classification_report(all_true_labels, all_predictions, target_names=["Safe (0)", "Injection (1)"]))

# ==========================================
# 9. VISUALIZE THE CONFUSION MATRIX (Optional but highly recommended)
# ==========================================
cm = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted Safe", "Predicted Injection"],
            yticklabels=["Actually Safe", "Actually Injection"])
plt.title("Guardrail Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()