import json
import os
from pathlib import Path

BASE_PATH = Path(os.getcwd())
DATA_PATH = BASE_PATH / 'Data' / 'Sentences'
CURRENT_PATH = Path(os.path.abspath(__file__)).parent

pretrains = ['pre_', '']
model_names = ['roberta', 'gpt2']
epochs = [1, 2, 4]
dataset_types = ['simple', 'variant']
reductions = ['red3', 'red10']

for pretrain in pretrains:
    for model_name in model_names:
        for epoch in epochs:
            for dataset_type in dataset_types:
                for reduction in reductions:
                    conf = {
                        "task": "nwp" if model_name=='gpt2' else "mlm",
                        "train_data_path": f'Data/Sentences/{dataset_type}/merge_{reduction}.txt', 
                        "eval_data_path": [
                            f'Data/Sentences/prompts/disease_drug_prompt_simple_{reduction}.csv',
                            f'Data/Sentences/prompts/drug_disease_prompt_simple_{reduction}.csv',
                            f'Data/Sentences/prompts/name_drug_prompt_simple_{reduction}.csv',
                            f'Data/Sentences/prompts/drug_name_prompt_simple_{reduction}.csv',
                            f'Data/Sentences/prompts/name_disease_prompt_simple_{reduction}.csv',
                            f'Data/Sentences/prompts/disease_name_prompt_simple_{reduction}.csv'
                        ],
                        "epochs" : epoch,
                        "model_name": "roberta-base" if model_name=='roberta' else 'gpt2',
                        "model_path": f"results/{pretrain}{model_name}" if pretrain else ''
                    }
                    if pretrain:
                        conf['model_name'] = ''

                    config_name = f"{pretrain}{model_name}_{epoch}_{dataset_type}_{reduction}.json"
                    with open(CURRENT_PATH/'finetune'/config_name, 'w') as f:
                        json.dump(conf, f)