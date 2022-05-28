import os, json, hashlib, pickle
from utils.data_utils import clinc_utils, snips_utils, banking77_utils, hwu64_utils

# some handy aliases
pjoin = os.path.join

write_json = lambda obj, path: json.dump(obj, open(path, "w"))
read_json = lambda path: json.load(open(path, "r"))

write_pickle = lambda obj, path: pickle.dump(obj, open(path, "wb"))
read_pickle = lambda path: pickle.load(open(path, "rb"))

write_file = lambda content, path: open(path, "w").write(content)
read_file = lambda path: open(path, "r").read()


def get_hash(x):
    return int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16)


def get_label_maps(data_root, data_name):
    with open(pjoin(data_root, data_name, "name2id.json"), "r") as f:
        name2id = json.load(f)
    with open(pjoin(data_root, data_name, "id2name.json"), "r") as f:
        id2name = json.load(f)
    return name2id, id2name


def get_ds_config(ds_name):
    # This code is used for "prepare_dataset.py"
    if ds_name == "clinc_oos":
        return clinc_utils.DS_CONFIG()
    elif ds_name == "snips_official":
        return snips_utils.DS_CONFIG()
    elif ds_name == "banking77":
        return banking77_utils.Banking77(ds_name)
    elif ds_name == "hwu64":
        return hwu64_utils.Hwu64(ds_name)
    else:
        raise NotImplementedError(f"Dataset {ds_name} not supported")


def truncate_data(data, num_examples):
    """Used to simulate the few-shot setting for validation intents."""
    return sorted(data, key=get_hash)[:num_examples]


def parse_and_load(dataset_name):
    # This code is used for "prepare_dataset.py"
    if dataset_name == "clinc_oos":
        return clinc_utils.parse_and_load_clinc()
    elif dataset_name == "snips_official":
        return snips_utils.parse_and_load_snips()
    elif dataset_name == "banking77":
        ds = banking77_utils.Banking77(dataset_name)
        return ds.parse_and_load()
    elif dataset_name == "hwu64":
        ds = hwu64_utils.Hwu64(dataset_name)
        return ds.parse_and_load()
    else:
        raise Exception(f"Dataset {dataset_name} not supported")
