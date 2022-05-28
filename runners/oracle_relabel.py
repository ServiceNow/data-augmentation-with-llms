from configs import exp_configs
import os, gc, argparse, pickle, torch, numpy as np
from utils import main_data_utils

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import DatasetDict, Dataset

from haven import haven_wizard as hw

torch.backends.cudnn.benchmark = True

pjoin = os.path.join
write_pickle = lambda obj, path: pickle.dump(obj, open(path, "wb"))
read_pickle = lambda path: pickle.load(open(path, "rb"))


def oracle_correction(exp_dict, savedir, args):
    """Main."""
    # ==========================
    # load full dataset (containing all generated examples)
    # ==========================
    ds_name = exp_dict["dataset"]["name"]
    ds_config = main_data_utils.get_ds_config(ds_name)
    DOMAINS = list(ds_config.domain_to_intent.keys())
    if "oos" in DOMAINS:
        print(f"ignoring OOS domain from {ds_name.upper()}")
        DOMAINS.pop(DOMAINS.index("oos"))  # remove oos

    print(f"{ds_name} domains:")
    print(DOMAINS)
    engines = ["ada", "babbage", "curie", "davinci", "gptj", "eda", "val", "test"]
    temp = [1.0]

    base_data_path = pjoin(args.datadir, ds_name, "full", "dataset.pkl")
    exp_data_path = pjoin(args.datadir, ds_name, "full", "data_full_suite.pkl")

    tokenizer = AutoTokenizer.from_pretrained(
        exp_dict["model"]["backbone"], use_fast=True
    )

    generated_dataset = DatasetDict(read_pickle(exp_data_path))

    # remove the domains and make a HF Dataset
    for e in engines:
        # no need to remove domains, will fetch from the base dataset
        if e in ["val", "test"]:
            continue
        elif e == "gptj":
            temp = [round(a, 1) for a in np.linspace(0.5, 2, int((2.1 - 0.5) / 0.1))]
        else:
            temp = [1.0]
        for t in temp:
            _lines = []
            _intents = []
            for d in DOMAINS:
                attr = e if e in ["eda", "bt"] else f"{e}_{t}"
                _lines.extend(generated_dataset[d]["F"][attr]["text"])
                _intents.extend(generated_dataset[d]["F"][attr]["intent"])
            generated_dataset[attr] = Dataset.from_dict(
                {"text": _lines, "intent": _intents}
            )

    base_dataset = read_pickle(base_data_path)
    # Add validation samples (for computing thresholds in fidelity plots)
    generated_dataset["val"] = Dataset.from_dict(base_dataset["val"])
    generated_dataset["test"] = Dataset.from_dict(base_dataset["test"])

    # ==========================
    # create model
    # ==========================
    print(f"Oracle path: {args.modeldir}")
    oracle = AutoModelForSequenceClassification.from_pretrained(
        args.modeldir, num_labels=exp_dict["dataset"]["num_labels"]
    )
    oracle.cuda()
    oracle.eval()

    # Init AL dataset
    al_path = pjoin(args.datadir, ds_name, "full", "al_dataset.pkl")
    if os.path.exists(al_path):
        print(f"Loading existing data from {al_path}")
        al_ds = read_pickle(al_path)["generated"]
    else:
        print(f"Initializing al_dataset.pkl for {ds_name.upper()}")
        al_ds = {}

    with torch.no_grad():
        for e in engines:
            if e == "gptj":
                temp = [
                    round(a, 1) for a in np.linspace(0.5, 2, int((2.1 - 0.5) / 0.1))
                ]
            else:
                temp = [1.0]
            for t in temp:
                attr = e if e in ["eda", "bt", "val", "test"] else f"{e}_{t}"
                if attr in al_ds:
                    print(f"{attr} already exists in {ds_name}'s AL dataset")
                    continue
                print(f"relabeling for {attr}")
                al_ds[attr] = {}
                al_ds[attr]["text"] = []
                al_ds[attr]["label"] = []

                encodings = tokenizer(
                    generated_dataset[attr]["text"],
                    max_length=50,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                total_num = len(encodings["input_ids"])
                batch_size = 100

                print("total_num", total_num, "and batch_size", batch_size)
                lbls = []
                for i in range(0, total_num, batch_size):
                    outputs = oracle(
                        encodings["input_ids"][i : i + 100].cuda(),
                        attention_mask=encodings["attention_mask"][i : i + 100].cuda(),
                    )
                    probs = torch.softmax(outputs.logits, dim=1)
                    lbls.extend(torch.argmax(probs, dim=1).cpu().tolist())

                al_ds[attr]["text"] = generated_dataset[attr]["text"]
                al_ds[attr]["intent"] = lbls
                al_ds[attr]["old_intent"] = generated_dataset[attr]["intent"]

    al_dataset = dict(
        train=base_dataset["train"],
        val=base_dataset["val"],
        test=base_dataset["test"],
        generated=al_ds,
    )

    with open(al_path, "wb") as f:
        pickle.dump(al_dataset, f)

    print("Emptying cache...")  # crucial!
    gc.collect()
    torch.cuda.empty_cache()


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
    parser.add_argument("-md", "--modeldir", type=str, default=None)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )

    args, unknown = parser.parse_known_args()
    if args.job_scheduler == "1":
        from configs import job_configs

        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None
    file_name = os.path.basename(__file__)[:-3]  # remove .py
    hw.run_wizard(
        func=oracle_correction,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_config,
        python_binary_path=args.python_binary,
        python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
    )
