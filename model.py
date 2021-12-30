from transformers import BertModel
import json
import os
import _jsonnet
import torch
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry


class Model:
    def __init__(self):
        BertModel.from_pretrained('bert-large-uncased-whole-word-masking')

        root_dir = '/content/rat-sql/'
        exp_config_path = root_dir + 'experiments/spider-bert-run.jsonnet'
        model_dir = root_dir + 'logdir/bert_run/bs=2,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,att=1/'
        checkpoint_step = 81000  # whatever checkpoint you want to use

        exp_config = json.loads(_jsonnet.evaluate_file(
            exp_config_path))  # data_path: '<path to spider/>',
        model_config_path = os.path.join(root_dir, exp_config["model_config"])
        model_config_args = exp_config.get("model_config_args")
        infer_config = json.loads(
            _jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

        self.inferer = Inferer(infer_config)
        self.inferer.device = torch.device("cpu")
        self.model = self.inferer.load_model(model_dir, checkpoint_step)
        self.dataset = registry.construct(
            'dataset', self.inferer.config['data']['val'])

        for _, schema in self.dataset.schemas.items():
            self.model.preproc.enc_preproc._preprocess_schema(schema)

    def infer(self, q, db_id):
        spider_schema = self.dataset.schemas[db_id]
        data_item = SpiderItem(
            text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
            code=None,
            schema=spider_schema,
            orig_schema=spider_schema.orig,
            orig={"question": q}
        )
        self.model.preproc.clear_items()
        enc_input = self.model.preproc.enc_preproc.preprocess_item(
            data_item, None)
        preproc_data = enc_input, None

        with torch.no_grad():
            return self.inferer._infer_one(self.model, data_item, preproc_data, beam_size=1, use_heuristic=True)


class ModelSingleton(object):
    _instance = None

    def __new__(Model):
        if Model._instance is None:
            print('Creating the object')
            Model._instance = super(ModelSingleton, Model).__new__(Model)
        return Model._instance
