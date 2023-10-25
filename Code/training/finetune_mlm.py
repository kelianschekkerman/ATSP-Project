# imports
import os, sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

# constants

DATA_OPTIONS = ['full', 'red3', 'red5', 'red10']
DATA_OPTION = DATA_OPTIONS[-1]

BASE_PATH = Path(os. getcwd())
DATA_PATH = BASE_PATH / 'Data' / 'Sentences'
EVAL_PATH = BASE_PATH / 'Data' / 'Preprocessed_data' / 'final'
NUM_TRAIN_EPOCHS = 2

MODEL_NAME = "allenai/biomed_roberta_base"
RUN_NAME = "biomed_roberta_finetuning",


# functions
def myprint(s):
    print(s.upper().center(20, '= '))


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myprint(f"Using device: {device}")

# Load your dataset
train_file = DATA_PATH / f'merge_{DATA_OPTION}.txt'
dataset = load_dataset('text', data_files={'train': train_file})

model_slug = MODEL_NAME.replace('/', '_')

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=32), batched=True)

# Prepare the MLM data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Load the model
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

output_dir = f"./results/{model_slug}/{NUM_TRAIN_EPOCHS}/{train_file.stem}"
print(f"### will save model at {output_dir}")

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,  # Change epochs as required
    logging_dir=f'./{output_dir}/logs',
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
    run_name=RUN_NAME,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



################ E V A L U A T I O N ################

# model = AutoModelForMaskedLM.from_pretrained('saved_model')
# tokenizer = AutoTokenizer.from_pretrained('saved_model')
model.eval()  # set the model to evaluation mode


def predict_masked_word(sentence, model, tokenizer):
    # Convert the sentence to token ids tensor
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1].item()
    
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits
    mask_token_logits = logits[0, mask_token_index, :]
    predicted_token_id = torch.argmax(mask_token_logits).item()

    return tokenizer.decode([predicted_token_id])


def compute_metrics(dataset, model, tokenizer):
    true_labels = []
    pred_labels = []

    for sentence_with_mask, true_word in dataset:
        predicted_word = predict_masked_word(sentence_with_mask, model, tokenizer)
        true_labels.append(true_word)
        pred_labels.append(predicted_word)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')

    return accuracy, f1, precision, recall

# Example usage:
# dataset = [("Hello, I am a [MASK] programmer.", "good"), ...]
# accuracy, f1, precision, recall = compute_metrics(dataset, model, tokenizer)

myprint("loading csv file".upper().center('20', '='))
eval_file = EVAL_PATH / f'name_disease_{DATA_OPTION}.csv'
data = pd.read_csv(eval_file)
sentences = []

for index, row in data.iterrows():
    sentence = f"Person {row['name']} has disease [MASK]."
    sentences.append((sentence, row['disease']))

    accuracy, f1, precision, recall = compute_metrics(sentences, model, tokenizer)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
