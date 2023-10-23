
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load tokenizer and model
model_name = "gpt2-medium"  # You can use "gpt2-small", "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="raw.txt",
    block_size=32
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define training arguments and train
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,  # You can increase the number of epochs based on your dataset
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
