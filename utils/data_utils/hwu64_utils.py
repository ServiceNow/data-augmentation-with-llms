"""
All HWU64 specific utilities are implemented here.
"""
from haven import haven_utils as hu
import os
import pandas as pd

INTENTS = [
    "alarm_query",
    "alarm_remove",
    "alarm_set",
    "audio_volume_down",
    "audio_volume_mute",
    "audio_volume_up",
    "calendar_query",
    "calendar_remove",
    "calendar_set",
    "cooking_recipe",
    "datetime_convert",
    "datetime_query",
    "email_addcontact",
    "email_query",
    "email_querycontact",
    "email_sendemail",
    "general_affirm",
    "general_commandstop",
    "general_confirm",
    "general_dontcare",
    "general_explain",
    "general_joke",
    "general_negate",
    "general_praise",
    "general_quirky",
    "general_repeat",
    "iot_cleaning",
    "iot_coffee",
    "iot_hue_lightchange",
    "iot_hue_lightdim",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "iot_hue_lightup",
    "iot_wemo_off",
    "iot_wemo_on",
    "lists_createoradd",
    "lists_query",
    "lists_remove",
    "music_likeness",
    "music_query",
    "music_settings",
    "news_query",
    "play_audiobook",
    "play_game",
    "play_music",
    "play_podcasts",
    "play_radio",
    "qa_currency",
    "qa_definition",
    "qa_factoid",
    "qa_maths",
    "qa_stock",
    "recommendation_events",
    "recommendation_locations",
    "recommendation_movies",
    "social_post",
    "social_query",
    "takeaway_order",
    "takeaway_query",
    "transport_query",
    "transport_taxi",
    "transport_ticket",
    "transport_traffic",
    "weather_query",
]


class Hwu64:
    def __init__(self, name):
        path_base = "./data/dialoglue/data_utils/dialoglue/hwu"

        self.data_name = name
        self.full_path = f"./data/{name}/full/dataset.pkl"
        self.num_examples = 10

        # 1. Get Intent Mappings
        name2id = {k: i for i, k in enumerate(INTENTS)}
        hu.save_json(f"./data/{name}/name2id.json", name2id)
        id2name = {i: k for i, k in enumerate(INTENTS)}
        hu.save_json(f"./data/{name}/id2name.json", id2name)

        # 2. save dataset.pkl
        if not os.path.exists(self.full_path):
            # get data dict
            data_dict = {
                "train": get_data_dict(path_base, "train", name2id),
                "val": get_data_dict(path_base, "val", name2id),
                "test": get_data_dict(path_base, "test", name2id),
            }
            hu.save_pkl(self.full_path, data_dict)
        else:
            data_dict = hu.load_pkl(self.full_path)

        # intents = list(name2id.keys())
        # Groupp by intent
        self.dataset_by_intent = {}
        for split in ["train", "val", "test"]:
            self.dataset_by_intent[split] = {}
            text_intent_dict = data_dict[split]
            text_list, intent_list = (
                text_intent_dict["text"],
                map(lambda x: id2name[x], text_intent_dict["intent"]),
            )

            # get texts from intent
            intent2texts = {}
            for t, i in zip(text_list, intent_list):
                if i not in intent2texts:
                    intent2texts[i] = []
                intent2texts[i] += [t]
            self.dataset_by_intent[split] = intent2texts

        self.domain_to_intent = self.generate_domain_to_intent_map()
        self.gpt3_batch_size: int = 128

    def parse_and_load(self):
        return self.dataset_by_intent

    def generate_domain_to_intent_map(self):
        mapping = {}
        for intent in INTENTS:
            # the intents are structured as {domain}_{intent}
            # E.g. the domain is alarm and intent is query for alarm_query
            domain = intent.split("_", 1)[0]
            if domain not in mapping:
                mapping[domain] = []
            mapping[domain].append(intent)
        return mapping


def get_data_dict(path_base, split, name2id):
    tmp_dict = pd.read_csv(os.path.join(path_base, f"{split}.csv")).to_dict()

    data_dict = {}
    data_dict["text"] = list(tmp_dict["text"].values())
    data_dict["intent"] = [int(name2id[c]) for c in tmp_dict["category"].values()]
    return data_dict
