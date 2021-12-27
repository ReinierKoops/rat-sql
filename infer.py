from transformers import BertModel
import json
import os
import _jsonnet
import torch
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry

BertModel.from_pretrained('bert-large-uncased-whole-word-masking')

root_dir = 'C:/Users/Shadow/Documents/GitHub/rat-sql/'
exp_config_path = root_dir + 'experiments/spider-bert-run.jsonnet'
model_dir = root_dir + 'logdir/bert_run/bs=2,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,att=1/'
checkpoint_step = 19400  # whatever checkpoint you want to use

exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))  # data_path: '<path to spider/>',
model_config_path = os.path.join(root_dir, exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

inferer = Inferer(infer_config)
inferer.device = torch.device("cuda")
model = inferer.load_model(model_dir, checkpoint_step)
dataset = registry.construct('dataset', inferer.config['data']['val'])

for _, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)


def question(q, db_id):
    spider_schema = dataset.schemas[db_id]
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=spider_schema,
        orig_schema=spider_schema.orig,
        orig={"question": q}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None

    with torch.no_grad():
        return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)

decoded, attention_map = question("Which employees are under age of 30 and which city do they work for?", "employee_hire_evaluation")

