import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
model = BertForMaskedLM.from_pretrained("prajjwal1/bert-small")

# 2. Tokenize the dataset
dataset = load_dataset('text', data_files={'train': 'raw.txt'})
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128), batched=True)

# 3. Prepare data for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# 4. Define training arguments and initialize Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100,
    output_dir="./bert_small_finetuned",
    evaluation_strategy="steps",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
)

# 5. Fine-tune the model
trainer.train()

# 6. Save the model
model.save_pretrained("./bert_small_finetuned")
tokenizer.save_pretrained("./bert_small_finetuned")
