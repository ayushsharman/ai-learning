# intent_detection.py
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

# Step 1: Data Preparation (Gather Materials)
texts = [
    "Is it sunny today?",          # weather_inquiry
    "Will it rain tomorrow?",       # weather_inquiry
    "Book a table for 2",           # book_restaurant
    "Reserve a seat tonight",       # book_restaurant
    "Play jazz music",              # play_music
    "Play classical songs",         # play_music
]
labels = [0, 0, 1, 1, 2, 2]  # 0=weather, 1=restaurant, 2=music

# Step 2: Tokenization (Blueprint)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Step 3: Model Setup (Foundation)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3  # 3 intents
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Dataset & Dataloader (Custom Rooms)
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=20,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset = IntentDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Training (Plumbing & Wiring)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Step 6: Prediction (Final Inspection)
def predict_intent(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
    
    intent_map = {
        0: "weather_inquiry",
        1: "book_restaurant",
        2: "play_music"
    }
    return intent_map[prediction]

# Test predictions
test_queries = [
    "Will it snow tonight?",
    "Reserve a table for 4 people",
    "Play rock music"
]

for query in test_queries:
    print(f"Query: '{query}' → Intent: {predict_intent(query)}")
