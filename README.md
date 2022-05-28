This is the official implementation of the following paper:<br>
[Gaurav Sahu](https://github.com/demfier), Pau Rodriguez, Issam Laradji, Parmida Atighehchian, David Vazquez, and Dzmitry Bahdanau. [Data Augmentation for Intent Classification with Off-the-shelf Large Language Models](https://aclanthology.org/2022.nlp4convai-1.5.pdf). *Proceedings of the 4th Workshop on NLP for Conversational AI, ACL 2022.*

If you find this code useful, please cite:
```bibtex
@inproceedings{sahu-etal-2022-data,
    title = "Data Augmentation for Intent Classification with Off-the-shelf Large Language Models",
    author = "Sahu, Gaurav  and
      Rodriguez, Pau  and
      Laradji, Issam  and
      Atighehchian, Parmida  and
      Vazquez, David  and
      Bahdanau, Dzmitry",
    booktitle = "Proceedings of the 4th Workshop on NLP for Conversational AI",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nlp4convai-1.5",
    pages = "47--57",
}
```

# Running experiments

### Datasets

#### Preparing data
This step is only required if there you are starting from scratch, i.e., *NO* data has been prepared at all.
Note that you are still required to create the symbolic link as suggested in the previous step.
To get started with data preparation, run the following:
```
python prepare_dataset.py --name <dataset_name> --data_root './data/'
```

This will generate samples for all the supported modes (upsample, gpt3, gptj, eda).
You can enable top-k and top-p sampling by specifying appropriate values for `--top_k` [0,) and `--top_p` [0, 1] flags.
**NOTE**: If generating for GPTJ, make sure there's enough GPU memory (recommended >=32G).

This will also setup the data directory structure for `<dataset_name>`.
It will prepare a `dataset.pkl` AND `data_full_suite.pkl`.
It will also generate the corresponding label maps (name2id, id2name).
Make sure you have `wget` installed in your local machine.

**Note:**
- HWU64 was downloaded from "https://github.com/alexa/dialoglue"
- Banking77, and CLINC150 were downloaded using the HuggingFace "datasets" library
- SNIPS was downloaded from "https://github.com/MiuLab/SlotGated-SLU"

Refer to the `fewshot_baseline_clinc` configuration in `configs/exp_configs/fs_exps.py` for full few-shot experiment config.

#### Running experiments:
To run baseline experiments following the original CLINC setting:
1. Edit the `baselines` variable inside `configs/exp_configs/fs_exps.py`. Here's an example for running `small` and `plus` baselines together:

```python
baselines = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": [
            {"name": "clinc_oos", "num_labels": 151, "config": c, "oos_id": 150}
            for c in ["plus", "small"]
        ],  # config: small/plus/full/few_pure
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": "baseline",  # intrinsic/baseline
        "lr": 4e-5,
        "batch_size": 32,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        "ngpu": 1,
        "eval_accumulation_steps": 30,
    }
)
```
**Note:** For CLINC (`name='clinc_oos'`), `oos_id=42` for `small/plus/imbalanced` and `oos_id=150` for `full/full_*`.
It also supports no-OOS classifiers. Set `oos_id=None` and `num_labels=150`.
For SNIPS (`name='snips_official'`) `oos_id=None` and it only supports `full*` settings. Make sure that `oos_id` is set correctly.
Refer to other config variables inside `configs/exp_configs.py` for partial few-shot (ex2 setup) and full few-shot configs.

3. Run experiments:
```
$(which python) -m runners.train --savedir_base /path/to/save/dir/ --exp_group_list baselines -j 1 -v results.ipynb --python_binary $(which python)
```
Setting `-j 0` will run it locally. `--exp_group_list ex2_setup` will run the EX2 experiments (make sure that the dataset preperation is complete)

#### For Oracle relabeling experiments:
  * Relabel generated examples using an oracle:
```
$(which python) -m runners.oracle_relabel -md /path/to/oracle/ -e fewshot_baseline_clinc
```
  * Train classifiers on the relabeled data:
```
$(which python) -m runners.train --savedir_base /path/to/save/dir/ --exp_group_list fewshot_oracle_clinc -j 1 -v results.ipynb --python_binary $(which python)
```
4. To compile results, correctly set the "knobs" in [runners.compile_results](runners/compile_results.py#L9-L30) and then run `python -m runners.compile_results` from root.


#### Adding a new dataset
To add a new dataset, follow these steps:
* Create a utils file for your dataset under `utils/data_utils/`. Let's call it `{dataset}_utils.py`.
All the dataset-specific processing needs to be added in there. In the end, your `{dataset}_utils.py` needs
to have a `parse_and_load_{dataset}` function. Refer to the documentation of `clinc_utils.parse_and_load_clinc()` to understand more.
* Add your dataset to `parse_and_load()` and `get_ds_config()` in `utils/data_utils/main.py`.
* [Running prepare_dataset.py](#preparing-data) for your dataset name now should create the required files for ex2 and non-ex2 setup.
* Finally, refer to [this](#for-oracle-relabling-experiments) to also generate dataset for the oracle relabling experiments in the full few-shot setup.


#### List of configs exp_configs.py for different experiments
1. Reproducing CLINC150 results:

```python
baselines = hu.cartesian_exp_group({
    # do multiple runs to account for stochasticity in metrics
    'run#': list(range(10)),
    'dataset': {'name': 'clinc_oos', 'num_labels': 151, 'oos_id': 150, 'config': 'full'},  # config: small/plus/full
    'model': {
        'name': 'intent_classification',
        'backbone':  'bert-large-uncased'
    },
    'exp_type': 'baseline',  # intrinsic/baseline
    'lr': 4e-5,
    'batch_size': 32,
    'epochs': 6,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    # metrics to compute
    'metrics': [['accuracy', 'f1', 'precision', 'recall']],
    'metric_best': 'accuracy',
    'ngpu': 1,
    'eval_accumulation_steps': 30
})
```

2. Running Partial few-shot baseline/upsample experiments:

```python
ex2_setup = hu.cartesian_exp_group({
    # do multiple runs to account for stochasticity in metrics
    'run#': list(range(10)),
    'dataset': [{
        'name': 'clinc_oos', 'num_labels': 151, 'oos_id': 150,
        'config': 'full_'+v} for v in DOMAINS],  # config -> small/imbalanced/plus/small_aug/full
    'model': {
        'name': 'intent_classification',
        'backbone':  'bert-large-uncased'
    },
    'exp_type': ['baseline', 'upsample'],  # gpt3/upsample/baseline
    'lr': 5e-5,
    'batch_size': 64,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    # metrics to compute. if oos_id is not None, 
    # compute inscope_accuracy and oos_recall as well
    'metrics': [['accuracy', 'f1', 'precision', 'recall']],
    'metric_best': 'f1',
    'eval_accumulation_steps': 30
})
```

3. Partial few-shot augmented (GPT3) experiments:

```python
ex2_setup = hu.cartesian_exp_group({
    # do multiple runs to account for stochasticity in metrics
    'run#': list(range(10)),
    'dataset': [{
        'name': 'clinc_oos', 'num_labels': 151, 'oos_id': 150,
        'config': 'full_'+v} for v in DOMAINS],  # config -> small/imbalanced/plus/small_aug/full
    'model': {
        'name': 'intent_classification',
        'backbone':  'bert-large-uncased'
    },
    'exp_type': ['gpt3'],  # gpt3/upsample/baseline
    'lr': 5e-5,
    'batch_size': 64,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    # metrics to compute. if oos_id is not None, 
    # compute inscope_accuracy and oos_recall as well
    'metrics': [['accuracy', 'f1', 'precision', 'recall']],
    'metric_best': 'f1',
    # 'gpt3_engine': 'ada',  # ada/babbage/curie/davinci
    'gpt3_engine': ['ada', 'babbage', 'curie', 'davinci'],  # ada/babbage/curie/davinci
    # 'gpt3_temp': 1.0,  # 0.5/0.6/0.7/0.8/0.9/1.0/1.5/2.0
    'gpt3_temp': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],  # 0.5-2.0
    'eval_accumulation_steps': 30
})
```

4. Training the oracle:

```python
baselines = hu.cartesian_exp_group({
    # do multiple runs to account for stochasticity in metrics
    'run#': list(range(1)),
    'dataset': {'name': 'clinc_oos', 'num_labels': 151, 'oos_id': None, 'config': 'full'},  # config: small/plus/full
    'model': {
        'name': 'intent_classification',
        'backbone':  'bert-large-uncased'
    },
    'exp_type': 'intrinsic',  # intrinsic/baseline
    'lr': 4e-5,
    'batch_size': 32,
    'epochs': 6,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    # metrics to compute
    'metrics': [['accuracy', 'f1', 'precision', 'recall']],
    'metric_best': 'accuracy',
    'ngpu': 1,
    'eval_accumulation_steps': 30
})
```
