
import os
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

BASE_PATH = Path(os. getcwd())
DATA_PATH = BASE_PATH / 'Data' / 'Sentences'
NUM_TRAIN_EPOCHS = 1

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "gpt2"  # "gpt2", "gpt2-medium", "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

raw_text_files = [
    DATA_PATH / 'merge_red3.csv',
    DATA_PATH / 'merge_red5.csv',
    DATA_PATH / 'merge_red10.csv',
    DATA_PATH / 'merge_full.csv',
]

for text_file in raw_text_files:
    output_dir = f"./results/{NUM_TRAIN_EPOCHS}/{text_file.stem}"
    print("start training".upper().center(20, '='))
    print(f">>> Filename: {text_file}")
    print(f">>> Model: {model_name}")
    print(f">>> Results: {output_dir}")
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
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=16,
        save_steps=100,
        save_total_limit=2,
        device=device
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
