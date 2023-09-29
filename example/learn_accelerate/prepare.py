"""
Provide some instance of model, optimizer and dataset
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

def get_model_optimizer_dataloader(batch_size = 8):
    dataset = load_dataset("yelp_review_full")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    # eval_dataset = dataset["test"].select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast = False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=False, load_from_cache_file = False, num_proc = 1)
    train_dataset = train_dataset.remove_columns(["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")
    #eval_dataset = eval_dataset.map(tokenize_function, batched=True, load_from_cache_file = False, num_proc = 1)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, train_dataloader

if __name__ == '__main__':
    get_model_optimizer_dataloader()
