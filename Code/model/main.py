import sys
import os
import json
from pathlib import Path

from finetune_nwp import train, predict
from eval import eval_predictions

if __name__ == "__main__":
    BASE_PATH = Path(os. getcwd()).parent.parent
    CURRENT_PATH = Path(os. getcwd())
    config_path = CURRENT_PATH / 'configs' / sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    output_dir = f"./results/{model_slug}/{NUM_TRAIN_EPOCHS}/{text_file.stem}"
    
    model, tokenizer = train(
        epochs=config['epochs'], 
        train_data_path=config['train_data_path'], 
        model_name=config['model_name'], 
        model_path=config['model_path'], 
        output_dir=output_dir
    )
    predictions, labels = predict(model, tokenizer, eval_data_path=config['eval_data_path'], output_dir=output_dir, n=10)
    
    # Evaluate full text and save to file
    eval_predictions(eval_data_path=config['eval_data_path'], predictions=predictions, labels=labels, output_dir=output_dir, operator='w', description="Evaluation with full predictions")

    # Evaluate only the first n characters
    n = 3           # Set the value for n 
    trunc_pred = [pred[:n] for pred in predictions]     # truncate predictions
    trunc_labels = [lbl[:n] for lbl in labels]        # truncate labels
    # Evaluate again with truncated data and append this to the file
    results_trunc = eval_predictions(eval_data_path=config['eval_data_path'], predictions=trunc_pred, labels=trunc_labels, output_dir=output_dir, operator='a', description="Evaluation with truncated data")
