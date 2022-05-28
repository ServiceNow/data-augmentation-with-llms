import argparse, numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import data_utils as du
from utils import main_data_utils as mdu


def compute_acc(preds, labels):
    return f"{np.mean([1 if p == l else 0 for p, l in zip(preds, labels)])*100:.2f}"


def get_seed_tuples(ds, seed_intent, ds_config, name2id):
    """
    returns seed sentences for a given intent_id in (full suite) dataset object
    ds as a list of tuples with (intent_id, sentence).
    seed_intent is a string
    """
    # identify the domain for the seed intent
    intent_id = int(name2id[seed_intent])
    seed_domain = None
    for _domain, _intents in ds_config.domain_to_intent.items():
        if seed_intent in _intents:
            seed_domain = _domain
            break

    # fetch the seed sentences for seed intent
    _start = ds[seed_domain]["F"]["train"]["intent"].index(intent_id)
    num_examples = ds[seed_domain]["F"]["train"]["intent"].count(intent_id)
    sents = ds[seed_domain]["F"]["train"]["text"][_start : _start + num_examples]
    return [(seed_intent, s) for s in sents]


def filter_via_gpt(sentences, seed_sentences, args):
    """
    `sentences` is a list of tuples (old intent, new intent, sentence)
    """
    # construct a prompt
    ltemplate = "sentence: {} ; category:{}"  # line template
    prompt = f"Each example in the following list contains a sentence that belongs to a category. A category is one of the following: {', '.join(args.intent_triplet)}"
    prompt += "\n\n"
    prompt += "\n".join([ltemplate.format(s, " " + i) for (i, s) in seed_sentences])
    prompt += "\n"

    print(f"FILTERING generations using {args.gpt_engine.upper()}...")
    retained_sentences = []
    # NOTE: ignoring the new intent (mid element of the tuple)
    for (old_intent, new_intent, sent) in tqdm(sentences):
        # query gpt
        input_prompt = prompt + ltemplate.format(sent, "")
        responses = du.augment_slices.openai_complete(
            prompt=input_prompt,
            n=10,
            engine=args.gpt_engine,
            temp=1.0,
            top_p=1.0,
        )
        responses = [r.text.strip() for r in responses]
        # print(input_prompt)
        # print(responses)
        # breakpoint()
        # NOTE: not handling ties. if there is a tie, it just means that
        # the sentence is not a good enough one, and we shouldn't include it.
        response = max(args.intent_triplet, key=responses.count)
        if response == old_intent:
            retained_sentences.append((old_intent, new_intent, sent))

    # num retained, num input
    n_r, n_i = len(retained_sentences), len(sentences)
    # percentage retained
    ret_per = f"{(n_r/n_i)*100:.2f}%"
    print(f"{args.gpt_engine.upper()} retained {n_r}/{n_i} sentences ({ret_per}).")
    return retained_sentences


def run_gpt_eval(eval_sentences, seed_sentences, args):
    """
    `eval_sentences` is a list of tuples (gt label in the dataset, text)
    `seed_sentences` is a list of tupels (seed intent, text)
    """
    # construct a prompt
    ltemplate = "sentence: {} ; category:{}"
    prompt = f"Each example in the following list contains a sentence that belongs to a category. A category is one of the following: {', '.join(args.intent_triplet)}"
    prompt += "\n\n"
    prompt += "\n".join([ltemplate.format(s, " " + i) for (i, s) in seed_sentences])
    prompt += "\n"

    preds, labels = [], []
    class_wise_pred_labels = defaultdict(list)
    print("Running evaluation...")
    for (intent, sent) in tqdm(eval_sentences):
        # query gpt
        input_prompt = prompt + ltemplate.format(sent, "")
        responses = du.augment_slices.openai_complete(
            prompt=input_prompt, n=10, engine=args.gpt_engine, temp=1.0, top_p=1.0
        )
        responses = [r.text.strip() for r in responses]
        # NOTE: not handling ties here.
        response = max(args.intent_triplet, key=responses.count)
        # for overall preds and labels
        preds.append(response)
        labels.append(intent)

        # for class-wise performance
        class_wise_pred_labels[intent].append(response)

    # Evaluation
    print(f"{args.gpt_engine.upper()} performance on {len(eval_sentences)} examples:")
    print(f"Overall accuracy = {compute_acc(preds, labels)}")
    print(f"Class-wise accuracies:")
    for intent, preds in class_wise_pred_labels.items():
        _acc = preds.count(intent) / len(preds)
        print(f"Acc. for intent {intent}: {_acc*100:.2f}")


def compute_fidelity(sentence_tuples):
    """
    `sentence_tuples` is a list of tuples (old intent, new intent, sentence)
    and old intent and new intent are intent names (not ids)
    """
    overall_3way_fidelity = []
    class_wise_fidelity = defaultdict(list)
    for (old_intent, new_intent, _) in sentence_tuples:
        class_wise_fidelity[old_intent].append(new_intent)
        overall_3way_fidelity.append(1 if new_intent == old_intent else 0)

    print(f"Overall 3-way fidelity = {np.mean(overall_3way_fidelity)*100:.2f}")
    for intent, preds in class_wise_fidelity.items():
        _fid = preds.count(intent) / len(preds)
        print(f"Fidelity for {intent}: {_fid*100:.2f}")


def oracle_eval(eval_sentences, id2name, name2id, args, size):
    """
    This will be used to evaluate:
    - small oracle (a.k.a 10-shot baseline)
    - bigger oracle (a.k.a the oracle)
    """

    intent_ids_of_interest = [name2id[i] for i in args.intent_triplet]
    intent_ids_of_interest.sort()

    # segregate sentences and labels
    labels, sentences = [], []
    for l, s in eval_sentences:
        labels.append(l)
        sentences.append(s)

    if args.dataset_name not in ["hwu64", "banking77"]:
        raise NotImplementedError(f"Dataset {args.dataset_name} not supported")

    print("Running N-shot baseline evaluation on reduced val+test set...")
    model_dir = f"/mnt/colab_public/results/few_shot_nlp/model/{args.dataset_name}/"
    model_dir += "10_shot_baseline/" if size == "small" else "oracle_checkpoint/"

    if args.dataset_name == "hwu64":
        num_labels = 64
    else:
        num_labels = 77

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        encodings = tokenizer(
            sentences,
            max_length=50,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        logits = model(input_ids.to(device), attention_mask.to(device)).logits
        # only consider the 3 intents of interest for prediction
        _temp_iids = torch.tensor(intent_ids_of_interest).to(device)
        logits = logits.index_select(index=_temp_iids, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        # remap pred ids to dataset intent ids
        preds = [id2name[str(intent_ids_of_interest[p])] for p in preds]

    class_wise_preds = defaultdict(list)
    for p, l in zip(preds, labels):
        class_wise_preds[l].append(p)

    # Evaluation
    print(f"N-shot baseline performance on {len(eval_sentences)} examples:")
    print(f"Overall accuracy = {compute_acc(preds, labels)}")
    print(f"Class-wise accuracies:")
    for intent, preds in class_wise_preds.items():
        _acc = preds.count(intent) / len(preds)
        print(f"Acc. for intent {intent}: {_acc*100:.2f}")


def main():
    """
    generated sentences
    filtered generated sentences

    val+test sentences for the dataset
    """
    args = parse_args()
    print(args)
    ds_config = mdu.get_ds_config(args.dataset_name)
    name2id = mdu.read_json(f"data/{args.dataset_name}/name2id.json")
    id2name = mdu.read_json(f"data/{args.dataset_name}/id2name.json")

    ############# START: Fetch all required set of sentences ################

    # fetch seed sentences for all the intents in the triplet using the
    # dataset prepared for partial fewshot experiment
    ds = mdu.read_pickle(f"data/{args.dataset_name}/full/data_full_suite.pkl")
    all_seed_sentences = []
    for intent in args.intent_triplet:
        seed_sents = get_seed_tuples(ds, intent, ds_config, name2id)
        all_seed_sentences.extend(seed_sents)
    # shuffle shuffle
    np.random.shuffle(all_seed_sentences)

    ds = mdu.read_pickle(f"data/{args.dataset_name}/full/al_dataset.pkl")
    # Fetch generated sentences as a list of tuples
    # each tuple --> (old intent, new intent, sentence)
    all_gen_sentences = []
    relabled_pool = ds["generated"][f"{args.gpt_engine}_1.0"]
    for idx, intent in enumerate(relabled_pool["old_intent"]):
        if id2name[str(intent)] not in args.intent_triplet:
            continue
        old_intent = id2name[str(intent)]
        new_intent = id2name[str(relabled_pool["intent"][idx])]
        sentence = relabled_pool["text"][idx]
        all_gen_sentences.append((old_intent, new_intent, sentence))

    # fetch val+test sentences (all accuracies are computed on these sentences)
    # NOTE: since it's val and test, old_intent is the ground truth
    eval_sentences = []
    for part in ["val", "test"]:
        sent_pool = ds["generated"][part]
        for (text, label) in zip(sent_pool["text"], sent_pool["old_intent"]):
            if id2name[str(label)] not in args.intent_triplet:
                continue
            eval_sentences.append((id2name[str(label)], text))

    # filtered examples using GPT as the classifier
    retained_generations = filter_via_gpt(all_gen_sentences, all_seed_sentences, args)

    ############## END: Fetched all required set of sentencs #################

    ############# COMPUTE DIFFERENT ##################
    print(f"\nFIDELITY for ALL {args.gpt_engine.upper()} generations")
    compute_fidelity(all_gen_sentences)
    print(f"\nFIDELITY after FILTERING {args.gpt_engine.upper()} rejections")
    compute_fidelity(retained_generations)

    print(f"\n{args.gpt_engine.upper()}'s 3-way classification performance")
    run_gpt_eval(eval_sentences, all_seed_sentences, args)
    print(f"\n10-SHOT-BASELINE's 3-way classification performance")
    oracle_eval(eval_sentences, id2name, name2id, args, "small")
    print(f"\nORACLE's 3-way classification performance")
    oracle_eval(eval_sentences, id2name, name2id, args, "big")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-dn", "--dataset-name", default="hwu64")
    parser.add_argument("-dn", "--dataset-name", default="banking77")
    parser.add_argument("-e", "--gpt_engine", default="davinci")
    parser.add_argument(
        "-it",
        "--intent-triplet",
        nargs="+",
        # default=["music_likeness", "play_music", "music_settings"],
        default=["topping_up_by_card", "top_up_failed", "pending_top_up"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
