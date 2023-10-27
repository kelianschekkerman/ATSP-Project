import csv
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

from utils import save_predictions, myprint

MASK_TOKEN = '<MASK>'

def train(epochs, train_data_path, model_name=None, model_path=None, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = train_data_path
    dataset = load_dataset('text', data_files={'train': str(train_file), 'eval': str(train_file)})

    tokenizer = AutoTokenizer.from_pretrained(model_name or model_path)
    tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained(model_name or model_path).to(device)

    print(f"### will save model at {output_dir}")

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=epochs,  # Change epochs as required
        logging_dir=str(output_dir/'logs'),
        output_dir=str(output_dir),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=1,
        warmup_steps=50,
        weight_decay=0.01,
        fp16=False,  # If you want to use mixed precision
        save_strategy="epoch",
        evaluation_strategy="epoch",
        run_name=output_dir.name,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
    )

    # Fine-tune the model
    myprint("training starts")
    trainer.train()

    # Save the model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return model, tokenizer


def generate_completions(sentence, model, tokenizer, n=10):
    # Convert the sentence to token ids tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence = sentence.replace(MASK_TOKEN, tokenizer.mask_token)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    input_ids = input_ids.to(device)
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1].item()
    
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits
    mask_token_logits = logits[0, mask_token_index, :]
    
    # Get the top 10 logits and their indices
    top_n_logits, top_n_indices = torch.topk(mask_token_logits, n)
    
    # Decode each of the top 10 token IDs
    predicted_tokens = [tokenizer.decode([idx.item()]) for idx in top_n_indices]

    return predicted_tokens


def predict(model, tokenizer, eval_data_path, output_dir, n=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # set the model to evaluation mode
    
    predictions = []
    labels = []

    print(">> Generating predictions...")
    with open(eval_data_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            completions = generate_completions(row['sentence'], model, tokenizer)
            predictions.append(completions)
            labels.append(row['label'])

    print(">> Saving and evaluations...")
    save_predictions(eval_data_path, predictions, labels, output_dir)
    return predictions, labels
