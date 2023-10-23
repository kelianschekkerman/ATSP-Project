
import os
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

BASE_PATH = Path(os. getcwd())
DATA_PATH = BASE_PATH / 'data' / 'sentences'
NUM_TRAIN_EPOCHS = 1

# Load tokenizer and model
model_name = "gpt2"  # "gpt2", "gpt2-medium", "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

raw_text_files = [
    DATA_PATH / 'person_to_drug_red3.csv',
    DATA_PATH / 'person_to_drug_red5.csv',
    DATA_PATH / 'person_to_drug_red10.csv',
    DATA_PATH / 'person_to_drug_full.csv',
]

for text_file in raw_text_files:
    print("start training".upper().center(20, '='))
    print(f">>> Filename: {text_file}")
    print(f">>> Model: {model_name}")
    print(f">>> Results: results/{text_file.stem}_{NUM_TRAIN_EPOCHS}")
    print()

    # Prepare dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=32
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define training arguments and train
    training_args = TrainingArguments(
        output_dir=f"./results/{text_file.stem}_{NUM_TRAIN_EPOCHS}",
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
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
