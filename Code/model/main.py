import sys
import os
import json
from pathlib import Path

import trainer_nwp, trainer_mlm
from eval import eval_predictions

if __name__ == "__main__":
    base_path = Path(os. getcwd()).parent.parent
    current_path = Path(os. getcwd())
    config_path = current_path / 'configs' / sys.argv[1]

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_slug = config['model_name'].split('/')[-1] if config['model_name'] else \
                 config['model_path'].split('/')[-1] + '_PRE_'
    output_dir = f"./results/{model_slug}_{config['epochs']}_{config['train_data_path'].stem}"

    trainer = trainer_mlm if config['task']=='mlm' else trainer_nwp

    print()
    print("start training".upper().center(40, '='))
    print(f">>> Train Dataset: {config['train_data_path']}")
    print(f">>> Model: {config['model_name'] or config['model_path']}")
    print(f">>> Will save at: {output_dir}")
    print()

    model, tokenizer = trainer.train(
        epochs=config['epochs'], 
        train_data_path=config['train_data_path'], 
        model_name=config['model_name'], 
        model_path=config['model_path'], 
        output_dir=output_dir
    )

    
    for eval_data_path in config['eval_data_path']:
        predictions, labels = trainer.predict(model, tokenizer, eval_data_path=eval_data_path, output_dir=output_dir, n=10)
        eval_predictions(eval_data_path=eval_data_path, predictions=predictions, labels=labels, output_dir=output_dir)

        # Evaluate full text and save to file
        eval_predictions(eval_data_path=config['eval_data_path'], predictions=predictions, labels=labels, output_dir=output_dir, operator='w', description="Evaluation with full predictions")

        # Evaluate only the first n characters
        n = 3           # Set the value for n 
        trunc_pred = [pred[:n] for pred in predictions]     # truncate predictions
        trunc_labels = [lbl[:n] for lbl in labels]        # truncate labels
        # Evaluate again with truncated data and append this to the file
        results_trunc = eval_predictions(eval_data_path=config['eval_data_path'], predictions=trunc_pred, labels=trunc_labels, output_dir=output_dir, operator='a', description="Evaluation with truncated data")

