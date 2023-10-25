import sys
import os
from pathlib import Path
import csv
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from eval import eval_predictions


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

    print()
    print("start training".upper().center(40, '='))
    print(f">>> Train Dataset: {train_data_path}")
    print(f">>> Eval Dataset: {eval_data_path}")
    print(f">>> Model: {model_name or model_path}")
    print(f">>> Will save at: {output_dir}")
    print()

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
        output_dir=output_dir,
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def generate_completions(prompt, model, tokenizer, n=10, max_length=10):
    """Generate n completions for a given prompt."""
    completions = []

    # Convert prompt to tensor input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for _ in range(n):
        # Generate completion
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_length, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95, do_sample=True)
        
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        completions.append(completion)

    return completions


def predict(model, tokenizer, eval_data_path, output_dir, n=10):
    predictions = []
    labels = []

    print(">> Generating predictions...")
    with open(eval_data_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(f"Sentence: {row['sentence']}, Label: {row['label']}")
            completions = generate_completions(row['sentence'], model, tokenizer)
            predictions.append(completions)
            labels.append(row['label'])

    print(">> Saving and evaluations...")
    save_predictions(eval_data_path, predictions, labels, output_dir)
    return predictions, labels


def save_predictions(eval_data_path, predictions, labels, output_dir):
    output_path = output_dir / f"pred_{eval_data_path.stem}.csv"
    predictions_count = len(predictions[0])
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header (optional)
        writer.writerow(["label"] + [str(i) for i in range(predictions_count)])
    
        # Write the data
        for label, prediction_list in zip(labels, predictions):
            writer.writerow([label] + prediction_list)
