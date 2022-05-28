from utils.data_utils.augment_slices import openai_complete
from utils.data_utils.main import read_pickle, pjoin, read_json, get_ds_config

K = 10
ENGINE = "curie"
DATASET_NAME = "clinc_oos"
INTENT_1, INTENT_2 = "user_name", "timer"
ID2NAME = read_json(pjoin("data", DATASET_NAME, "id2name.json"))
NAME2ID = read_json(pjoin("data", DATASET_NAME, "name2id.json"))
DS = read_pickle(pjoin("data", DATASET_NAME, "full", "data_full_suite.pkl"))


def get_seed_samples(intent_name):
    seed_domain = None
    for domain, intent_list in get_ds_config(DATASET_NAME).domain_to_intent.items():
        if intent_name in intent_list:
            seed_domain = domain
            break
    _start = DS[seed_domain]["F"]["train"]["intent"].index(NAME2ID[intent_name])
    return DS[seed_domain]["F"]["train"]["text"][_start : _start + K]


def main():
    lines1 = get_seed_samples(INTENT_1)
    lines2 = get_seed_samples(INTENT_2)
    prompt = f"The following sentences belong to the same category {INTENT_2}:\n"
    prompt += "\n".join([f"Example {i+1}: {l}" for i, l in enumerate(lines2)])
    prompt += "\n\n"
    prompt += f"The following sentences belong to the same category {INTENT_1}:\n"
    prompt += "\n".join([f"Example {i+1}: {l}" for i, l in enumerate(lines1)])
    prompt += "\n"
    prompt += f"Example {K+1}:"
    print(prompt)
    from pprint import pprint

    pprint(
        [
            r.text.strip()
            for r in openai_complete(
                prompt=prompt, n=20, engine=ENGINE, temp=1.0, top_p=1.0
            )
        ]
    )

    print("----Old method----")
    prompt = f"The following sentences belong to the same category {INTENT_1}:\n"
    prompt += "\n".join([f"Example {i+1}: {l}" for i, l in enumerate(lines1)])
    prompt += "\n"
    prompt += f"Example {K+1}:"
    print(prompt)
    from pprint import pprint

    pprint(
        [
            r.text.strip()
            for r in openai_complete(
                prompt=prompt, n=20, engine=ENGINE, temp=1.0, top_p=1.0
            )
        ]
    )


if __name__ == "__main__":
    main()
