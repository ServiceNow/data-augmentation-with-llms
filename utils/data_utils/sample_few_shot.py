"""
This script samples K(default=10) exemplers per intent class in the few-shot
slice using the heuristic used in the EX2 paper:
https://arxiv.org/pdf/2102.01335.pdf

Since we will do cross-validation, every domain in the SNIPS dataset will be
treated as a few-shot slice once.
See https://github.com/clinc/oos-eval/blob/master/supplementary.pdf.
"""
import os
import pickle

from utils.data_utils.main import get_label_maps, truncate_data, parse_and_load

pjoin = os.path.join


def add_fs_slice(data_splits, val_domain, data_root, ds_config):
    DOMAIN_TO_INTENT = ds_config.domain_to_intent
    few_shot_intents = set(DOMAIN_TO_INTENT[val_domain])

    # load data grouped by intent, will also prepare the basic dataset.pkl
    data = parse_and_load(ds_config.data_name)
    name2id, id2name = get_label_maps(data_root, ds_config.data_name)

    # NOTE: `data` has oos_train, oos_val, and oos_test keys as well.
    # ignoring them here. since we don't perform a ex2-setup run for the OOS
    # domain, OOS samples are added when required in data_loader.py
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # construct D_M and D_F (with respective, train, val, test sets)
    # MANY SHOT INTENTS
    # -----------------
    m_train_lines, m_train_intents = [], []
    for intent, lines in train_data.items():
        if intent not in few_shot_intents:
            m_train_lines.extend(lines)
            m_train_intents.extend([name2id[intent]] * len(lines))

    m_val_lines, m_val_intents = [], []
    for intent, lines in val_data.items():
        if intent not in few_shot_intents:
            m_val_lines.extend(lines)
            m_val_intents.extend([name2id[intent]] * len(lines))

    m_test_lines, m_test_intents = [], []
    for intent, lines in test_data.items():
        if intent not in few_shot_intents:
            m_test_lines.extend(lines)
            m_test_intents.extend([name2id[intent]] * len(lines))

    # FEW SHOT INTENTS
    # ----------------
    f_train_lines, f_train_intents = [], []
    for intent, lines in train_data.items():
        if intent in few_shot_intents:
            f_train_lines.extend(truncate_data(lines, ds_config.num_examples))
            f_train_intents.extend([name2id[intent]] * ds_config.num_examples)

    f_val_lines, f_val_intents = [], []
    for intent, lines in val_data.items():
        if intent in few_shot_intents:
            f_val_lines.extend(lines)
            f_val_intents.extend([name2id[intent]] * len(lines))

    f_test_lines, f_test_intents = [], []
    for intent, lines in test_data.items():
        if intent in few_shot_intents:
            f_test_lines.extend(lines)
            f_test_intents.extend([name2id[intent]] * len(lines))

    data_splits[val_domain] = {}

    # add the many-shot split
    data_splits[val_domain]["M"] = {
        "train": {
            "text": m_train_lines,
            "intent": m_train_intents,
        },
        "val": {
            "text": m_val_lines,
            "intent": m_val_intents,
        },
        "test": {
            "text": m_test_lines,
            "intent": m_test_intents,
        },
    }
    # add the few-shot split
    data_splits[val_domain]["F"] = {
        "train": {
            "text": f_train_lines,
            "intent": f_train_intents,
        },
        "val": {
            "text": f_val_lines,
            "intent": f_val_intents,
        },
        "test": {
            "text": f_test_lines,
            "intent": f_test_intents,
        },
    }


def sample_few_shot(data_root, ds_config):
    save_path = pjoin(data_root, ds_config.data_name, "full", "data_full_suite.pkl")
    if os.path.exists(save_path):
        print(f"Sample few shot has already been called for {ds_config.data_name}")
        return
    data_splits = {}
    DOMAIN_TO_INTENT = ds_config.domain_to_intent.keys()

    for val_domain in DOMAIN_TO_INTENT:
        # in-place modifies data_splits
        add_fs_slice(data_splits, val_domain, data_root, ds_config)

    # note that this pkl file will be updated
    with open(save_path, "wb") as f:
        pickle.dump(data_splits, f)
    return data_splits
