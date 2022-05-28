from collections import defaultdict
import gc, os, GPUtil, torch
from transformers import AutoTokenizer, TrainingArguments, Trainer

from utils.metrics import Metrics
from utils import main_data_utils as mdu
from utils.data_utils.data_loader import DatasetLoader
from models.backbones import get_backbone


torch.backends.cudnn.benchmark = True


def trainval(exp_dict, savedir, cl_args):
    """Main."""
    # ==========================
    # load datasets
    # ==========================
    dataset_loader = DatasetLoader(cl_args.datadir, exp_dict)

    # ==========================
    # create model and trainer
    # ==========================
    backbone = get_backbone(exp_dict)

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

    if cl_args.job_scheduler == "1":
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
        save_total_limit=1,
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
    trainer.train()
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
    # write results
    results_path = os.path.join(savedir, "code/results.json")
    # this will happen if not using scheduler
    if not os.path.exists(results_path):
        os.makedirs(os.path.dirname(results_path))
    mdu.write_json({exp_dict["dataset"]["config"]: metrics}, results_path)

    if not cl_args.retain_checkpoints:
        # delete HF checkpoints to save space on toolkit
        print(f"DELETING! {os.path.join(savedir, 'checkpoint-*')}")
        os.system(f"rm -rf {os.path.join(savedir, 'checkpoint-*')}")
