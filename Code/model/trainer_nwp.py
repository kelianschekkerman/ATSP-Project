import sys
import os
from pathlib import Path
import csv
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm

from utils import save_predictions

MASK_TOKEN = '<MASK>'

def train(epochs, train_data_path, model_name=None, model_path=None, output_dir=None):
    """
    If model name is None it will load a pretrained model from model_path
    train_data_path is a raw text file.
    eval_data_path is a pd file that has an incomplete sentence in first row and \
    an actual label in the second row
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

    # Prepare dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_data_path,
        block_size=32
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define training arguments and train
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        save_steps=100,
        save_total_limit=2
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
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return model, tokenizer


def generate_completions(sentences, model, tokenizer, n=10, max_length=5):
    """Generate n completions for a given prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = len(sentences)
    # Preprocess
    prompts = [sentence.split(MASK_TOKEN)[0] for sentence in sentences]
    input_ids = tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_length, num_return_sequences=n, temperature=1.0, top_k=50, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    completions = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    completions = [completions[i][len(prompts[i]):] for i in range(len(prompts))]
    completions = [' '.join(c.replace(',', ' ').replace('.', ' ').split()[:2]) for c in completions]
    completions = [completions[i:i+n] for i in range(0, len(completions), batch_size)]
    return completions


def predict(model, tokenizer, eval_data_path, output_dir, n=10, batch_size=16):
    predictions = []
    labels = []

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(">> Generating predictions...")
    with open(eval_data_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        sentences_batch, labels_batch = [], []
        for row in tqdm(reader):
            sentences_batch.append(row['sentence'])
            labels_batch.append(row['label'])

            if len(sentences_batch) == batch_size:
                completions = generate_completions(sentences_batch, model, tokenizer)
                predictions.extend(completions)
                labels.extend(labels_batch)
                sentences_batch, labels_batch = [], []

    print(">> Saving and evaluations...")
    save_predictions(eval_data_path, predictions, labels, output_dir)
    return predictions, labels
