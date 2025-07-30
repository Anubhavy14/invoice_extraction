import os
import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "layoutlmv3-invoice-model"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
EPOCHS = 3
MAX_LENGTH = 512

# Label mapping
LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-HEADER", "I-HEADER"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Initialize processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

class FUNSDDataset(Dataset):
    def __init__(self, annotations, image_dir, processor, max_length=MAX_LENGTH):
        self.annotations = annotations
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.image_dir, item["image_path"])
        image = Image.open(image_path).convert("RGB")
        
        # Extract words and boxes from either top-level or nested 'words'
        words = []
        boxes = []
        word_labels = []
        
        for element in item["words"]:
            # Handle both direct elements and nested 'words' arrays
            if "words" in element:
                for word in element["words"]:
                    words.append(word["text"])
                    boxes.append(word["box"])
                    word_labels.append(LABEL2ID[word.get("label", "O")])
            else:
                words.append(element["text"])
                boxes.append(element["box"])
                word_labels.append(LABEL2ID[element.get("label", "O")])
        
        # Process with padding and truncation
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=False
        )
        
        # Convert all to tensors
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding['labels'] = encoding['labels'].to(torch.long)
        
        return encoding

def load_funsd_dataset(base_dir="D:/master"):
    """Load dataset with comprehensive validation"""
    annotations = []
    image_dir = os.path.join(base_dir, "images")
    annotation_dir = os.path.join(base_dir, "annotations")
    
    print("\nScanning dataset...")
    print(f"Image directory: {image_dir}")
    print(f"Annotation directory: {annotation_dir}")
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.json')]
    print(f"Found {len(annotation_files)} annotation files")
    
    for ann_file in annotation_files:
        base_name = os.path.splitext(ann_file)[0]
        image_path = f"{base_name}.png"  # Assuming PNG format
        
        try:
            with open(os.path.join(annotation_dir, ann_file), "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Validate JSON structure
                if "form" not in data:
                    print(f"Skipping {ann_file}: Missing 'form' field")
                    continue
                
                # Flatten all words from form elements
                all_words = []
                for element in data["form"]:
                    if "words" in element:
                        all_words.extend(element["words"])
                    else:
                        all_words.append(element)
                
                # Validate word elements
                valid_words = []
                for word in all_words:
                    if not all(k in word for k in ["text", "box"]):
                        print(f"Skipping invalid word in {ann_file}")
                        continue
                    if len(word["box"]) != 4:
                        print(f"Skipping word with invalid box in {ann_file}")
                        continue
                    valid_words.append(word)
                
                if not valid_words:
                    print(f"Skipping {ann_file}: No valid words found")
                    continue
                
                annotations.append({
                    "image_path": image_path,
                    "words": valid_words
                })
                
        except json.JSONDecodeError as e:
            print(f"Skipping {ann_file}: Invalid JSON ({str(e)})")
        except Exception as e:
            print(f"Skipping {ann_file}: {str(e)}")
    
    print(f"\nSuccessfully loaded {len(annotations)} valid annotation-image pairs")
    return annotations, image_dir

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'bbox': torch.stack([item['bbox'] for item in batch]),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def train():
    try:
        print("\n=== Starting Training ===")
        annotations, image_dir = load_funsd_dataset()
        
        if not annotations:
            raise ValueError("\nNo valid annotations found. Please check:")
        
        print(f"\nDataset successfully loaded with {len(annotations)} samples")
        train_ann, val_ann = train_test_split(annotations, test_size=0.2, random_state=42)
        
        print("\nCreating datasets...")
        train_dataset = FUNSDDataset(train_ann, image_dir, processor)
        val_dataset = FUNSDDataset(val_ann, image_dir, processor)
        
        print("\nCreating dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        
        print("\nInitializing model...")
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        ).to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        
        print("\nStarting training loop...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                    outputs = model(**inputs)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        print("\nSaving model...")
        model.save_pretrained(MODEL_PATH)
        processor.save_pretrained(MODEL_PATH)
        print(f"\n=== Training complete! Model saved to {MODEL_PATH} ===")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify all JSON files contain a 'form' array")
        print("2. Each form element should have:")
        print("   - 'text' (string)")
        print("   - 'box' (list of 4 numbers)")
        print("   - Optional 'label' field")
        print("3. Check all image files are in PNG format")
        print("4. Ensure base filenames match exactly (e.g., 0001.png and 0001.json)")

if __name__ == "__main__":
    train()