from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

BASE_PATH = Path(os. getcwd())
DATA_PATH = BASE_PATH / 'Data' / 'Sentences'
NUM_TRAIN_EPOCHS = 2

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your dataset
dataset = load_dataset('text', data_files={'train': 'path_to_your_text_file.txt'})

model_name = "allenai/biomed_roberta_base"
model_slug = model_name.replace('/', '_')

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512), batched=True)

# Prepare the MLM data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)


raw_text_files = [
    DATA_PATH / 'merge_red3.csv',
    DATA_PATH / 'merge_red5.csv',
    DATA_PATH / 'merge_red10.csv',
    DATA_PATH / 'merge_full.csv',
]

text_file = raw_text_files[2]
output_dir = f"./results/{model_slug}/{NUM_TRAIN_EPOCHS}/{text_file.stem}"

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
    run_name="biomed_roberta_finetuning",
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
