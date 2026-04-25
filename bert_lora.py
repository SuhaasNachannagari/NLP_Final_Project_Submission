import pandas as pd
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm

# Map labels to integers
label_mapping = {'neutral': 0, 'banter': 1, 'toxic': 2}
reverse_mapping = {0: 'neutral', 1: 'banter', 2: 'toxic'}

# Function to load and preprocess a dataframe
def load_and_preprocess_df(file_path):
    df = pd.read_csv(file_path)
    if 'message' not in df.columns or 'new_label' not in df.columns:
        raise ValueError(f"'{file_path}' must contain 'message' and 'new_label' columns.")
    df = df.dropna(subset=['message', 'new_label'])
    df['label'] = df['new_label'].map(label_mapping).astype(int)
    return df

# 1. Load Data
print("Loading training, validation, and test data...")
train_df = load_and_preprocess_df('training.csv')
val_df = load_and_preprocess_df('validation.csv')
test_df = load_and_preprocess_df('test.csv')

train_texts = train_df['message'].tolist()
train_labels = train_df['label'].tolist()
val_texts = val_df['message'].tolist()
val_labels = val_df['label'].tolist()
test_texts = test_df['message'].tolist()
test_labels = test_df['label'].tolist()

print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}, Test size: {len(test_texts)}")

# 2. Tokenization & Dataset
MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 128
BATCH_SIZE = 16

class DotaChatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = DotaChatDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = DotaChatDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = DotaChatDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3. Model, Loss, Optimizer Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=5e-5)
EPOCHS = 10

# 4. Training Loop
train_losses = []
val_losses = []
best_macro_f1 = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs.logits, labels)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    val_preds, val_true = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_val_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    macro_f1 = f1_score(val_true, val_preds, average='macro')
    print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {macro_f1:.4f}')

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_model_state = copy.deepcopy(model.state_dict())
        print("New best model found! Saving checkpoint...")

model.load_state_dict(best_model_state)
model.save_pretrained('./best_bert_lora')
tokenizer.save_pretrained('./best_bert_lora')
print(f'\nTraining complete! Best Val Macro F1: {best_macro_f1:.4f}')

# 5. Evaluation & Visualization

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('bert_lora_loss.png', dpi=150, bbox_inches='tight')
print("Loss plot saved to bert_lora_loss.png")

# Test set evaluation
model.eval()
test_preds, test_true = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

ordered_labels = ['banter', 'neutral', 'toxic']
ordered_int_labels = [label_mapping[label] for label in ordered_labels]

print("\n--- Test Set Classification Report ---")
print(classification_report(test_true, test_preds, target_names=ordered_labels))

# Confusion matrix
cm = confusion_matrix(test_true, test_preds, labels=ordered_int_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ordered_labels, yticklabels=ordered_labels)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('DOTA 2 Classification: Actual vs. Predicted')
plt.savefig('bert_lora_confusion.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved to bert_lora_confusion.png")