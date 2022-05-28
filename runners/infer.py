import os

os.environ["WANDB_DISABLED"] = "true"

from collections import defaultdict
import argparse
from haven import haven_wizard as hw
import gc
import json
import GPUtil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from utils.metrics import Metrics
from utils.data_utils.data_loader import DatasetLoader


torch.backends.cudnn.benchmark = True

NUM_LABELS = 7
DATASET_NAME = "snips_official"
MODEL_DIR = (
    f"/mnt/colab_public/results/few_shot_nlp/model/{DATASET_NAME}/oracle_checkpoint/"
)

RESULTS_PATH = f"results/gpt_fidelity_{DATASET_NAME}.json"


def write_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f)


def trainval(exp_dict, savedir, args):
    """Main."""
    # ==========================
    # load datasets
    # ==========================
    dataset_loader = DatasetLoader(args.datadir, exp_dict)

    # ==========================
    # create model and trainer
    # ==========================
    backbone = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=NUM_LABELS
    )

    tokenizer = AutoTokenizer.from_pretrained(
        exp_dict["model"]["backbone"], use_fast=True
    )

    def preprocess(example):
        # note that setting max_length and truncation will not
        # have any effect for the vanilla baseline experiments
        results = tokenizer(example["text"], max_length=50, truncation=True)
        results["label"] = example["intent"]
        return results

    encoded_dataset = dataset_loader.dataset.map(preprocess, batched=True)

    if args.job_scheduler == "1":
        from configs import job_configs

        n_gpu = job_configs.JOB_CONFIG["resources"]["gpu"]
    else:
        n_gpu = 1

    args = TrainingArguments(
        savedir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=exp_dict["lr"],
        per_device_train_batch_size=exp_dict["batch_size"] // n_gpu,
        per_device_eval_batch_size=exp_dict["batch_size"] // n_gpu,
        num_train_epochs=exp_dict["epochs"],
        warmup_ratio=exp_dict["warmup_ratio"],
        weight_decay=exp_dict["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model=exp_dict["metric_best"],
        # push_to_hub=True,
        # push_to_hub_model_id=f"{model_name}-finetuned-{task}",
    )

    if "full_validation" in encoded_dataset:
        # for ex2 setup experiments
        eval_split = "full_validation"
    else:
        # for clinc setup experiments
        eval_split = "validation"

    trainer = Trainer(
        backbone,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[eval_split],
        tokenizer=tokenizer,
        compute_metrics=Metrics(exp_dict).compute_metrics(),
    )
    # trainer.train()
    print(GPUtil.showUtilization())

    print(f"n_gpus: {trainer.args.n_gpu}, local rank: {trainer.args.local_rank}")
    print("Emptying cache...")  # crucial!
    gc.collect()
    torch.cuda.empty_cache()

    # compute metrics
    trainer.args.eval_accumulation_steps = exp_dict["eval_accumulation_steps"]
    if "full_test" in encoded_dataset:
        metrics = {"overall": {}, "few_shot": {}}
    else:
        metrics = defaultdict(dict)

    for split in encoded_dataset:
        if "train" in split:
            continue

        # split test, full_test, validation, full_validation
        prefix = "test" if "test" in split else "valid"
        if "overall" in metrics:
            _type = "overall" if "full" in split else "few_shot"
            metrics[_type].update(
                trainer.evaluate(encoded_dataset[split], metric_key_prefix=prefix)
            )
        elif exp_dict["exp_type"] == "intrinsic":
            metrics[split].update(
                trainer.evaluate(encoded_dataset[split], metric_key_prefix=prefix)
            )
        else:
            metrics.update(
                trainer.evaluate(encoded_dataset[split], metric_key_prefix=prefix)
            )
    # print results
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group_list",
        nargs="+",
        default="resnet",
        help="name of an experiment in exp_configs.py",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/haven_output",
        help="folder where logs will be saved",
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-d", "--datadir", type=str, default="./data")
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Overwrite previous results"
    )
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument("-v", default="results.ipynb", help="orkestrator")
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, unknown = parser.parse_known_args()
    from configs import exp_configs

    if args.job_scheduler == "1":
        from configs import job_configs

        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    file_name = os.path.basename(__file__)[:-3]  # remove .py
    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_config,
        python_binary_path=args.python_binary,
        python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
    )
