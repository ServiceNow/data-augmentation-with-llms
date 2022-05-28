# prepare SNIPS dataset for intent classification
import os
import json
import pickle
import collections
from typing import Dict
from pydantic import BaseModel


class DS_CONFIG(BaseModel):
    data_name: str = "snips_official"
    full_path: str = "./data/snips_official/full/dataset.pkl"
    num_examples: int = 10
    domain_to_intent: Dict = {
        "AddToPlaylist": ["AddToPlaylist"],
        "BookRestaurant": ["BookRestaurant"],
        "GetWeather": ["GetWeather"],
        "PlayMusic": ["PlayMusic"],
        "RateBook": ["RateBook"],
        "SearchCreativeWork": ["SearchCreativeWork"],
        "SearchScreeningEvent": ["SearchScreeningEvent"],
    }
    gpt3_batch_size: int = 128


SNIPS_DIR = './data/snips_official/'


def make_label_maps():
    with open(os.path.join(SNIPS_DIR, 'full', 'train', 'label')) as f:
        labels = sorted(set([x.strip() for x in f.readlines()]))
    intent_name2id = {label: idx for idx, label in enumerate(labels)}
    intent_id2name = {idx: label for idx, label in enumerate(labels)}
    with open(os.path.join(SNIPS_DIR, 'name2id.json'), 'w') as f:
        json.dump(intent_name2id, f)
    with open(os.path.join(SNIPS_DIR, 'id2name.json'), 'w') as f:
        json.dump(intent_id2name, f)


def download_snips_full():
    # NOTE: the official SNIPS repo uses valid, train, and test as split names
    for split in ['train', 'valid', 'test']:
        download_snips_files(split)

def download_snips_files(split):
    print(f'Downloading SNIPS {split} files')
    snips_base_url = f'https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/{split}/'
    download_dir = os.path.join(SNIPS_DIR, 'full', split)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    os.system(f'wget {snips_base_url + "seq.in"} -P {download_dir}')
    os.system(f'wget {snips_base_url + "label"} -P {download_dir}')


def parse_snips_split(dataset, split, name2id_map):
    with open(os.path.join(SNIPS_DIR, 'full', split, 'seq.in')) as f:
        sentences = [x.strip() for x in f.readlines()]
    with open(os.path.join(SNIPS_DIR, 'full', split, 'label')) as f:
        labels = [name2id_map[x.strip()] for x in f.readlines()]

    # just trying to be consistent with naming splits across diff datasets
    dataset['val' if split == 'valid' else split] = {
        'text': sentences,
        'intent': labels
    }


def prepare_snips():
    download_snips_full()
    make_label_maps()

    dataset = {}
    name2id_map = json.load(open(os.path.join(SNIPS_DIR, 'name2id.json')))
    for split in ['train', 'valid', 'test']:
        parse_snips_split(dataset, split, name2id_map)

    with open(os.path.join(SNIPS_DIR, 'full', 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print('Base dataset.pkl prepared for SNIPS')


def check_snips():
    with open(os.path.join(SNIPS_DIR, 'full', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    print(len(dataset['train']['text']))
    print(len(dataset['val']['text']))
    print(len(dataset['test']['text']))


def load_snips():
    print('Loading SNIPS dataset')
    id2name = json.load(open(os.path.join(SNIPS_DIR, 'id2name.json')))
    data_full_path = DS_CONFIG().full_path
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(data_full_path, 'rb') as data_file:
        for split_name, split_data in pickle.load(data_file).items():
            for query, intent in zip(split_data['text'], split_data['intent']):
                intent = id2name[str(intent)]
                data[split_name][intent].append(query)
    return data


def parse_and_load_snips():
    if not os.path.exists(os.path.join(SNIPS_DIR, 'full', 'dataset.pkl')):
        prepare_snips()
    check_snips()
    return load_snips()
