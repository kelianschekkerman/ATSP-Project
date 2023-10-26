import sys
import os
import json
from pathlib import Path

import trainer_nwp, trainer_mlm
from eval import eval_predictions

if __name__ == "__main__":
    base_path = Path(os. getcwd())
    config_path = base_path / 'Code' / 'model' / 'configs' / sys.argv[1]

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_slug = config['model_name'].split('/')[-1] if config['model_name'] else \
                 config['model_path'].split('/')[-1] + '_PRE_'
    output_dir = Path(f"results/{model_slug}_{config['epochs']}_{Path(config['train_data_path']).stem}")

    trainer = trainer_mlm if config['task']=='mlm' else trainer_nwp

    print()
    print("start training".upper().center(40, '='))
    print(f">>> Train Dataset: {config['train_data_path']}")
    print(f">>> Model: {config['model_name'] or config['model_path']}")
    print(f">>> Will save at: {output_dir}")
    print()

    model, tokenizer = trainer.train(
        epochs=config['epochs'], 
        train_data_path=Path(config['train_data_path']), 
        model_name=config['model_name'], 
        model_path=Path(config['model_path']), 
        output_dir=output_dir
    )
    
    for _eval_data_path in config['eval_data_path']:
        eval_data_path = Path(_eval_data_path)
        predictions, labels = trainer.predict(model, tokenizer, eval_data_path=eval_data_path, output_dir=output_dir, n=10)
        eval_predictions(eval_data_path=eval_data_path, predictions=predictions, labels=labels, output_dir=output_dir)
