"""
All CLINC specific utilities must be implemented here.
It defines a DS_CONFIG for the dataset.
The main function that will be used by any other submodule in the repo is
`parse_and_load_clinc`. In a nutshell, it prepares a `dataset.pkl` file for
CLINC if not already prepared AND returns the dataset grouped by intent names.
Refer to the function's documentation to know more details.
"""

import pickle
import os
import json
import collections
from typing import Dict
from pydantic import BaseModel


class DS_CONFIG(BaseModel):
    data_name: str = "clinc_oos"
    full_path: str = "./data/clinc_oos/full/dataset.pkl"
    num_examples: int = 10
    # See https://github.com/clinc/oos-eval/blob/master/supplementary.pdf.
    domain_to_intent: Dict = {
        "banking": [
            "transfer",
            "transactions",
            "balance",
            "freeze_account",
            "pay_bill",
            "bill_balance",
            "bill_due",
            "interest_rate",
            "routing",
            "min_payment",
            "order_checks",
            "pin_change",
            "report_fraud",
            "account_blocked",
            "spending_history",
        ],
        "credit_card": [
            "credit_score",
            "report_lost_card",
            "credit_limit",
            "rewards_balance",
            "new_card",
            "application_status",
            "card_declined",
            "international_fees",
            "apr",
            "redeem_rewards",
            "credit_limit_change",
            "damaged_card",
            "replacement_card_duration",
            "improve_credit_score",
            "expiration_date",
        ],
        "dining": [
            "recipe",
            "restaurant_reviews",
            "calories",
            "nutrition_info",
            "restaurant_suggestion",
            "ingredients_list",
            "ingredient_substitution",
            "cook_time",
            "food_last",
            "meal_suggestion",
            "restaurant_reservation",
            "confirm_reservation",
            "how_busy",
            "cancel_reservation",
            "accept_reservations",
        ],
        "home": [
            "shopping_list",
            "shopping_list_update",
            "next_song",
            "play_music",
            "update_playlist",
            "todo_list",
            "todo_list_update",
            "calendar",
            "calendar_update",
            "what_song",
            "order",
            "order_status",
            "reminder",
            "reminder_update",
            "smart_home",
        ],
        "auto": [
            "traffic",
            "directions",
            "gas",
            "gas_type",
            "distance",
            "current_location",
            "mpg",
            "oil_change_when",
            "oil_change_how",
            "jump_start",
            "uber",
            "schedule_maintenance",
            "last_maintenance",
            "tire_pressure",
            "tire_change",
        ],
        "travel": [
            "book_flight",
            "book_hotel",
            "car_rental",
            "travel_suggestion",
            "travel_alert",
            "travel_notification",
            "carry_on",
            "timezone",
            "vaccines",
            "translate",
            "flight_status",
            "international_visa",
            "lost_luggage",
            "plug_type",
            "exchange_rate",
        ],
        "utility": [
            "time",
            "alarm",
            "share_location",
            "find_phone",
            "weather",
            "text",
            "spelling",
            "make_call",
            "timer",
            "date",
            "calculator",
            "measurement_conversion",
            "flip_coin",
            "roll_dice",
            "definition",
        ],
        "work": [
            "direct_deposit",
            "pto_request",
            "taxes",
            "payday",
            "w2",
            "pto_balance",
            "pto_request_status",
            "next_holiday",
            "insurance",
            "insurance_change",
            "schedule_meeting",
            "pto_used",
            "meeting_schedule",
            "rollover_401k",
            "income",
        ],
        "small_talk": [
            "greeting",
            "goodbye",
            "tell_joke",
            "where_are_you_from",
            "how_old_are_you",
            "what_is_your_name",
            "who_made_you",
            "thank_you",
            "what_can_i_ask_you",
            "what_are_your_hobbies",
            "do_you_have_pets",
            "are_you_a_bot",
            "meaning_of_life",
            "who_do_you_work_for",
            "fun_fact",
        ],
        "meta": [
            "change_ai_name",
            "change_user_name",
            "cancel",
            "user_name",
            "reset_settings",
            "whisper_mode",
            "repeat",
            "no",
            "yes",
            "maybe",
            "change_language",
            "change_accent",
            "change_volume",
            "change_speed",
            "sync_device",
        ],
        "oos": ["oos"],
    }
    gpt3_batch_size: int = 128


DOMAIN_TO_INTENT = DS_CONFIG().domain_to_intent
CLINC_FULL_PATH = "./data/clinc_oos/full/data_full.json"


def make_label_maps():
    """
    returns a mapping of intent name to intent id and vice versa
    """
    intent_names = []
    for intent_list in DOMAIN_TO_INTENT.values():
        if intent_list == ["oos"]:
            continue
        intent_names.extend(intent_list)

    name2id, id2name = {}, {}
    for idx, name in enumerate(sorted(set(intent_names))):
        name2id[name] = idx
        id2name[idx] = name

    # explicitly assign oos an id of 150
    name2id["oos"] = 150
    id2name[150] = "oos"

    # storing in JSON for readability later on
    # NOTE: keys in id2name.json will be str and not int!
    with open("./data/clinc_oos/name2id.json", "w") as f:
        json.dump(name2id, f)
    with open("./data/clinc_oos/id2name.json", "w") as f:
        json.dump(id2name, f)
    return name2id, id2name


def download_clinc_full():
    download_dir = "./data/clinc_oos/full/"
    # create dir if doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    clinc_data_url = (
        "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
    )
    os.system(f"wget {clinc_data_url} -P {download_dir}")
    return json.load(open(os.path.join(download_dir, "data_full.json")))


def load_or_download_clinc():
    try:
        full_data = json.load(open(CLINC_FULL_PATH))
    except FileNotFoundError:
        full_data = download_clinc_full()
    return full_data


def get_label_maps():
    try:
        intentname2id = json.load(open("./data/clinc_oos/name2id.json"))
        intentid2name = json.load(open("./data/clinc_oos/id2name.json"))
    except FileNotFoundError:
        intentname2id, intentid2name = make_label_maps()
    return intentname2id, intentid2name


def prepare_clinc():
    full_data = load_or_download_clinc()
    intentname2id, intentid2name = get_label_maps()

    data_dict = {
        "train": {"text": [], "intent": []},
        "val": {"text": [], "intent": []},
        "test": {"text": [], "intent": []},
    }

    for key, data in full_data.items():
        for sample in data:
            line, label = sample
            label = intentname2id[label]
            # maybe there's a better way to prepare this...
            if "train" in key:
                data_dict["train"]["text"].append(line)
                data_dict["train"]["intent"].append(label)
            elif "val" in key:
                data_dict["val"]["text"].append(line)
                data_dict["val"]["intent"].append(label)
            else:
                data_dict["test"]["text"].append(line)
                data_dict["test"]["intent"].append(label)

    print("Data details:")
    print(
        f'Train #lines: {len(data_dict["train"]["text"])} #labels: {len(data_dict["train"]["intent"])}'
    )
    print(
        f'Val #lines: {len(data_dict["val"]["text"])} #labels: {len(data_dict["val"]["intent"])}'
    )
    print(
        f'Test #lines: {len(data_dict["test"]["text"])} #labels: {len(data_dict["test"]["intent"])}'
    )

    with open("./data/clinc_oos/full/dataset.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print("Base dataset.pkl prepared for CLINC!")


def load_clinc():
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(CLINC_FULL_PATH, "r") as data_file:
        for split_name, split_data in json.load(data_file).items():
            for query, intent in split_data:
                data[split_name][intent].append(query)
    return data


def parse_and_load_clinc():
    """
    This functions has two primary roles:
    1) create `dataset.pkl` for CLINC if it doesn't exist already
    2) return the CLINC dataset grouped by intent names

    Secondary role:
    This function also creates name2id.json and id2name.json files for the
    dataset in the respective dataset folder (e.g. ./data/clinc_oos/ for
    CLINC. Note that the name of the dataset folder matches
    DS_CONFIG.data_name). If the dataset contains an OOS class, make it the
    last class for that dataset (to easen execution of non-OOS experiments)

    name2id.json is a Dict[Str: Int] and id2name.json is a Dict[Str: Str]

    parse_and_load_clinc() is the only function that will interact with the
    outside world, and any dataset specific utility required to accomplish the
    described roles above (like downloading, parsing, etc.) may be implemented
    in this file.

    NOTE that we store intent ids in dataset.pkl, but the grouped dataset
    returned by the function stores intent names!

    `dataset.pkl` contains a Dict object with the following schema:

    {
        'train': {'text': listofStr, 'intent': listofInt},
        'val': {'text': listofStr, 'intent': listofInt},
        'test': {'text': listofStr, 'intent': listofInt}
    }

    Format of the returned dataset grouped by intent names:

    collections.defaultdict(None, {
        'train': {
            Str: listofStr,
            Str: listofStr,
            .
            .
            .
            Str: listofStr  # n_intents
        },

        'val': {
            Str: listofStr,
            Str: listofStr,
            .
            .
            .
            Str: listofStr  # n_intents
        },

        'test': {
            Str: listofStr,
            Str: listofStr,
            .
            .
            .
            Str: listofStr  # n_intents
        }
    })

    parse_and_load_clinc: None -> collections.defaultdict
    """
    if not os.path.exists(os.path.join("./data/clinc_oos/full/dataset.pkl")):
        prepare_clinc()
    return load_clinc()
