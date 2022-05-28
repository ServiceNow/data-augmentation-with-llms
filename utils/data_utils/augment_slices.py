"""
This script uses openai to augment the dataset with
samples for different few-shot domains
"""
import os, gc, torch, openai, pickle, json
import numpy as np
from . import eda_utils
from collections import Counter

pjoin = os.path.join


class GPTJChoice:
    def __init__(self, text):
        self.text = text


def load_dataset_slices(data_root, data_name):
    with open(pjoin(data_root, data_name, "full", "data_full_suite.pkl"), "rb") as f:
        return pickle.load(f)


def openai_complete(
    prompt,
    n,
    engine,
    temp,
    top_p,
    max_tokens=32,
    stop="\n",
    frequency_penalty=0,
    logprobs=None,
):
    completion = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temp,
        top_p=1 if not top_p else top_p,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
    )
    return completion.choices
    # return [c.text.strip() for c in completion.choices]


def gptj_complete(prompt, n, temp, model, tokenizer, top_k, top_p):
    """
    Parameters:
    ===========
    prompt: Str
        Text to be fed as prompt
    n: Int
        Number of sentences to be generated
    temp: Float
        Sampling temperature for GPTJ
    model: GPTJ model instance
        GPTJ model loaded for inference
    tokenizer: GPTJ tokenizer instance
        GPTJ tokenizer loaded (from Huggingface currenlty)
    top_k: Anyof False, Int
        top k tokens to consider when sampling
    top_p: Float
        p value for top-p sampling (nucleus sampling)
    """
    # k is the line where the predicted/generated sample resides
    # compensate for the last (incomplete) line "Example {num_seed+1}:"
    k = len(prompt.splitlines()) - 1
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    stop_token = tokenizer.encode("\n")[0]
    sentences = []
    while len(sentences) != n:
        # generate multiple sentences at a time
        # NOTE: .generate already sets torch.no_grad()
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            max_length=2 * input_ids.shape[1],
            temperature=temp,
            eos_token_id=stop_token,
            # 30 is the max we can go on a 32G GPU
            # ^^that's a lie
            num_return_sequences=min(n, 30),
            # to suppress open-end generation warning
            pad_token_id=stop_token,
            top_k=0 if not top_k else top_k,
            top_p=1 if not top_p else top_p,
        )
        generations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        del gen_tokens
        # remove the first k lines as they belong to the prompt
        for i in range(min(n, 30)):
            # intentionally using that space after :
            # the model should be predicting that
            s = generations[i].splitlines()[k:][0][len(f"Example {k-1}: ") :].strip()
            # don't add if empty or if already generated n sentences
            if s and len(sentences) < n:
                sentences.append(s)
            del s
        del generations
    del input_ids
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return [GPTJChoice(s) for s in sentences]


def upsample_domain(prompt, n):
    lines = prompt.strip().splitlines()
    upsampled = lines * (n // len(lines))
    upsampled.extend(lines[: (n % len(lines))])
    return upsampled


def eda_domain(prompt, n):
    lines = prompt.strip().splitlines()
    k = len(lines)
    augmented = []
    for line in lines:
        if not line:
            continue
        # augment for a line
        # NOTE: num_aug in the EDA paper is #new lines per training sample
        # using alpha = 0.05 as per recommendation in the paper
        generated = eda_utils.eda(
            sentence=line,
            alpha_sr=0.05,
            alpha_ri=0.05,
            alpha_rs=0.05,
            p_rd=0.05,
            num_aug=n // k,
        )
        augmented.extend(generated)
    return augmented


def regenerate(input_prompt, n_empty, engine, temp, top_p):
    new_lines = []
    while n_empty > 0:
        print(f"Saw {n_empty} empty line(s). GPT3ing again...")
        curr_lines = openai_complete(
            prompt=input_prompt,
            n=n_empty,
            engine=engine,
            temp=temp,
            top_p=top_p,
        )
        curr_lines = [r.text.strip() for r in curr_lines]
        n_empty = curr_lines.count("")
        new_lines.extend([t for t in curr_lines if t])
        if n_empty == 0:
            return new_lines


def augment_domain(
    dataset_slices,
    val_domain,
    data_save_path,
    id2name,
    num_ex=10,
    n_max=128,
    engine=None,
    temp=None,
    model=None,
    tokenizer=None,
    top_k=False,
    top_p=False,
    mode="upsample",
    mt_dict=None,
):
    """
    Augments a given domain in dataset_slices AND updates the pickle file
    """
    if len(dataset_slices[val_domain]["M"]["train"]["intent"]) == 0:
        # no many-domain available for this dataset
        data_path = os.path.join(os.path.dirname(data_save_path), "dataset.pkl")
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        num_synthetic = int(
            np.median(list(Counter(dataset["train"]["intent"]).values()))
        )
    else:
        # when many-domain is available
        counts = Counter(dataset_slices[val_domain]["M"]["train"]["intent"])
        num_synthetic = int(np.median(list(counts.values())))

    f_train_lines = dataset_slices[val_domain]["F"]["train"]["text"]
    f_train_labels = dataset_slices[val_domain]["F"]["train"]["intent"]

    f_gen_lines, f_gen_labels = [], []
    for idx in range(0, len(f_train_lines), num_ex):
        prompt_lines = f_train_lines[idx : idx + num_ex]

        # simple prompt format (prepend 'Example i: ')
        input_prompt = "\n".join(
            [f"Example {i+1}: {t}" for i, t in enumerate(prompt_lines)]
        )
        input_prompt += f"\nExample {num_ex+1}:"

        # prompting with label addition as well
        seed_intent = id2name[f"{f_train_labels[idx : idx + num_ex][0]}"]
        print(f"Seed intent: {seed_intent}")
        input_prompt = (
            f"The following sentences belong to the same category '{seed_intent}':\n"
            + input_prompt
        )

        if mode == "upsample":
            print("Upsampling...")
            generated_lines = upsample_domain(input_prompt, num_synthetic)
        elif mode == "eda":
            print("EDAing...")
            generated_lines = eda_domain(input_prompt, num_synthetic)
        elif mode == "gptj":
            engine = "gptj"
            print("GPTJing...")
            generated_lines = gptj_complete(
                prompt=input_prompt,
                n=num_synthetic,
                temp=temp,
                model=model,
                tokenizer=tokenizer,
                top_k=top_k,
                top_p=top_p,
            )
            generated_lines = [r.text.strip() for r in generated_lines]
        else:
            print("GPT3ing...")
            if num_synthetic <= n_max:
                generated_lines = openai_complete(
                    prompt=input_prompt,
                    n=num_synthetic,
                    engine=engine,
                    temp=temp,
                    top_p=top_p,
                )
                generated_lines = [r.text.strip() for r in generated_lines]
            else:
                generated_lines = []
                for _ in range(num_synthetic // n_max):
                    _c = openai_complete(
                        prompt=input_prompt,
                        n=n_max,
                        engine=engine,
                        temp=temp,
                        top_p=top_p,
                    )
                    generated_lines.extend([r.text.strip() for r in _c])
                # rest of the lines
                _c = openai_complete(
                    prompt=input_prompt,
                    n=num_synthetic % n_max,
                    engine=engine,
                    temp=temp,
                    top_p=top_p,
                )
                generated_lines.extend([r.text.strip() for r in _c])

            # sometimes there are empty strings generated by GPT3, try again
            n_empty = generated_lines.count("")
            if n_empty > 0:
                generated_lines = [t for t in generated_lines if t]
                if n_empty <= n_max:
                    generated_lines.extend(
                        regenerate(
                            input_prompt,
                            n_empty,
                            engine,
                            temp,
                            top_p,
                        )
                    )
                else:
                    for _ in range(n_empty // n_max):
                        generated_lines.extend(
                            regenerate(
                                input_prompt,
                                n_max,
                                engine,
                                temp,
                                top_p,
                            )
                        )
                    # rest of the lines
                    generated_lines.extend(
                        regenerate(
                            input_prompt,
                            n_empty % n_max,
                            engine,
                            temp,
                            top_p,
                        )
                    )

            assert len(generated_lines) == num_synthetic

        f_gen_lines.extend(generated_lines)
        # using len(generated_lines) to make sure #lines == #labels
        # as, for imbalanced datasets, there can be slightly more sentences
        # generated by EDA as it augments per sentence.
        f_gen_labels.extend([f_train_labels[idx]] * len(generated_lines))

    attr_name = mode if engine is None else f"{engine}_{temp}"

    dataset_slices[val_domain]["F"][attr_name] = {
        "text": f_gen_lines,
        "intent": f_gen_labels,
    }
    write_pickle(data_save_path, dataset_slices)


def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def upsample_loop(dataset_slices, domains, data_save_path, id2name):
    # Upsample loop
    for val_domain in domains:
        if "upsample" in dataset_slices[val_domain]["F"].keys():
            print(f"upsample for {val_domain} already exists")
            continue
        print(f"Augmenting for domain: {val_domain}")
        augment_domain(
            dataset_slices=dataset_slices,
            val_domain=val_domain,
            data_save_path=data_save_path,
            id2name=id2name,
        )


def eda_loop(dataset_slices, domains, data_save_path, id2name):
    """
    Easy data augmenatation baseline by Wei and Zhou (EMNLP, 2019)
    """
    # EDA loop:
    for val_domain in domains:
        if "eda" in dataset_slices[val_domain]["F"].keys():
            print(f"eda for {val_domain} already exists")
            continue
        print(f"Augmenting for domain: {val_domain}")
        augment_domain(
            dataset_slices=dataset_slices,
            val_domain=val_domain,
            data_save_path=data_save_path,
            id2name=id2name,
            mode="eda",
        )


def load_gptj():
    """returns GPTJ and its tokenizer"""
    from transformers import GPTJForCausalLM, AutoTokenizer

    print("Loading GPT-J...")
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to("cuda")
    model.eval()
    print("Loaded GPT-J.")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return model, tokenizer


def gptj_loop(
    dataset_slices,
    domains,
    ds_config,
    data_save_path,
    id2name,
    top_k,
    top_p,
):
    # GPTJ loop
    model, tokenizer = None, None
    # for temp in [1.0]:
    # round the temperatures to avoid floating points as 1.200003..
    for temp in [round(a, 1) for a in np.linspace(0.5, 2, int((2.1 - 0.5) / 0.1))]:
        print(f"Engine: GPT-J | Temp: {temp}")
        for val_domain in domains:
            # comment the following two lines to *update* existing lines
            if f"gptj_{temp}" in dataset_slices[val_domain]["F"].keys():
                print(f"gptj_{temp} for {val_domain} already exists")
                continue

            if model is None and tokenizer is None:
                model, tokenizer = load_gptj()

            print(f"Augmenting for domain: {val_domain}")
            augment_domain(
                dataset_slices=dataset_slices,
                val_domain=val_domain,
                data_save_path=data_save_path,
                id2name=id2name,
                num_ex=ds_config.num_examples,
                engine="gptj",
                temp=temp,
                model=model,
                tokenizer=tokenizer,
                top_k=top_k,
                top_p=top_p,
                mode="gptj",
            )


def gpt3_loop(
    dataset_slices,
    domains,
    ds_config,
    data_save_path,
    id2name,
    top_p,
):
    # GPT3 loop
    for engine in ["davinci", "curie", "babbage", "ada"]:
        for temp in [1.0]:
            # round the temperatures to avoid floating points as 1.200003..
            # for temp in [round(a, 1) for a in np.linspace(0.5, 2, int((2.1 - 0.5) / 0.1))]:
            print(f"Engine: {engine} | Temp: {temp}")
            for val_domain in domains:
                # comment the following two lines to *update* existing lines
                if f"{engine}_{temp}" in dataset_slices[val_domain]["F"].keys():
                    print(f"{engine}_{temp} for {val_domain} already exists")
                    continue
                print(f"Augmenting for domain: {val_domain}")
                augment_domain(
                    dataset_slices=dataset_slices,
                    val_domain=val_domain,
                    data_save_path=data_save_path,
                    id2name=id2name,
                    num_ex=ds_config.num_examples,
                    n_max=ds_config.gpt3_batch_size,
                    engine=engine,
                    temp=temp,
                    top_p=top_p,
                    mode="gpt3",
                )
                # davinci quickly reaches the token/min limit, so we must sleep
                if engine == "davinci":
                    print("sleeping, for openai won't let me GPT3 no more...")
                    import time

                    time.sleep(60)


def augment_slices(
    data_root,
    ds_config,
    modes=["upsample", "gptj", "gpt3", "eda"],
    top_k=False,
    top_p=False,
):
    dataset_slices = load_dataset_slices(data_root, ds_config.data_name)
    DOMAINS = ds_config.domain_to_intent.keys()

    data_save_path = pjoin(
        data_root,
        ds_config.data_name,
        "full",
        "data_full_suite.pkl",
    )
    id2name = json.load(open(pjoin(data_root, ds_config.data_name, "id2name.json")))

    for mode in modes:
        if mode == "upsample":
            upsample_loop(dataset_slices, DOMAINS, data_save_path, id2name)
        elif mode == "gptj":
            gptj_loop(
                dataset_slices,
                DOMAINS,
                ds_config,
                data_save_path,
                id2name,
                top_k=top_k,
                top_p=top_p,
            )
        elif mode == "gpt3":
            if top_k:
                print("NOTE: ignoring top_k for gpt3 as openai doesn't support it yet")
            gpt3_loop(
                dataset_slices,
                DOMAINS,
                ds_config,
                data_save_path,
                id2name,
                top_p=top_p,
            )
        elif mode == "eda":
            eda_loop(dataset_slices, DOMAINS, data_save_path, id2name)
    return dataset_slices
