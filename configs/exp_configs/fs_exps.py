from haven import haven_utils as hu

EXP_GROUPS = {}

# CLINC
DOMAINS = [
    "banking",
    "credit_card",
    "dining",
    "home",
    "auto",
    "travel",
    "utility",
    "work",
    "small_talk",
    "meta",
]


baselines = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": [
            {"name": "clinc_oos", "num_labels": 151, "config": c, "oos_id": 150}
            for c in ["few_pure"]
        ],  # config: small/plus/full/few_pure
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": "baseline",  # intrinsic/baseline
        "lr": 4e-5,
        "batch_size": 32,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        "ngpu": 1,
        # 'gpt3_engine': ['ada', 'babbage', 'curie', 'davinci'],
        # 'gpt3_temp': 1.0,
        "eval_accumulation_steps": 30,
    }
)

ex2_setup = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": [
            {
                "name": "clinc_oos",
                "num_labels": 151,
                "oos_id": 150,
                "config": "full_" + v,
            }
            for v in DOMAINS
        ],  # config -> small/imbalanced/plus/small_aug/full
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": ["gpt3"],  # gpt3/upsample/baseline
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute. if oos_id is not None,
        # compute inscope_accuracy and oos_recall as well
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        # 'gpt3_engine': 'ada',  # ada/babbage/curie/davinci
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci
        # 'gpt3_temp': 1.0,  # 0.5/0.6/0.7/0.8/0.9/1.0/1.5/2.0
        "gpt3_temp": 1.0,  # 0.5-2.0
        "eval_accumulation_steps": 30,
    }
)

fewshot_oracle_clinc = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "clinc_oos",
            "num_labels": 151,
            "oos_id": 150,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3_oracle",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_gpt3_clinc = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "clinc_oos",
            "num_labels": 151,
            "oos_id": 150,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_gpt3mix_clinc = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "clinc_oos",
            "num_labels": 151,
            "oos_id": 150,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": ["gpt3mix", "gpt3mix_oracle"],
        "soft_label": True,
        "gpt3_engine": "curie",  # only generated for curie
        "gpt3_temp": 1.0,  # only generated for 1.0
        "eval_accumulation_steps": 30,
    }
)

fewshot_eda_clinc = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "clinc_oos",
            "num_labels": 151,
            "oos_id": 150,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": ["eda", "eda_oracle"],  # eda/eda_oracle
        "eval_accumulation_steps": 30,
    }
)


fewshot_baseline_clinc = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "clinc_oos",
            "num_labels": 151,
            "oos_id": 150,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,  # 1e-5 for k=3, 5
        "batch_size": 8,  # 2 for k=3, 5
        "epochs": 6,  # 20 for k=3, 5
        "warmup_ratio": 0.1,
        "weight_decay": 0.001,  # 0.0001 for k=3, 5
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "baseline",
        "eval_accumulation_steps": 30,
    }
)

banking77_baselines = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": {
            "name": "banking77",
            "num_labels": 77,
            "config": "full",
            "oos_id": None,
        },  # config: small/plus/full/few_pure
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": "baseline",  # intrinsic/baseline
        "lr": 5e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        "eval_accumulation_steps": 30,
    }
)

fewshot_baseline_banking77 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "banking77",
            "num_labels": 77,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 8,
        "epochs": 20,
        "warmup_ratio": 0.1,
        "weight_decay": 0.001,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "baseline",
        "eval_accumulation_steps": 30,
    }
)

fewshot_oracle_banking77 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "banking77",
            "num_labels": 77,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3_oracle",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_gpt3_banking77 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "banking77",
            "num_labels": 77,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        # "gpt3_engine": "curie",  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)


fewshot_eda_banking77 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "banking77",
            "num_labels": 77,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": ["eda", "eda_oracle"],
        "eval_accumulation_steps": 30,
    }
)

hwu64_baselines = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": {
            "name": "hwu64",
            "num_labels": 64,
            "config": "full",
            "oos_id": None,
        },  # config: small/plus/full/few_pure
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": "intrinsic",  # intrinsic/baseline
        "lr": 5e-5,
        "batch_size": 32,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        "eval_accumulation_steps": 30,
    }
)

fewshot_baseline_hwu64 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "hwu64",
            "num_labels": 64,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 8,
        "epochs": 20,
        "warmup_ratio": 0.1,
        "weight_decay": 0.001,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "baseline",
        "eval_accumulation_steps": 30,
    }
)

fewshot_oracle_hwu64 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "hwu64",
            "num_labels": 64,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3_oracle",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        # "gpt3_engine": "babbage",  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,
        "eval_accumulation_steps": 30,
    }
)

fewshot_gpt3_hwu64 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "hwu64",
            "num_labels": 64,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        # "gpt3_engine": "babbage",  # ada/babbage/curie/davinci/gptj
        # "gpt3_temp": [round(a, 1) for a in np.linspace(0.5, 2, int((2.1 - 0.5) / 0.1))],
        "gpt3_temp": [1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_eda_hwu64 = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "hwu64",
            "num_labels": 64,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 32,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": ["eda", "eda_oracle"],
        "eval_accumulation_steps": 30,
    }
)

# SNIPS
SNIPS_DOMAINS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]

# SNIPS_DOMAINS = ["RateBook"]

snips_baselines = hu.cartesian_exp_group(
    {
        "dataset": {
            "name": "snips_official",
            "num_labels": 7,
            "oos_id": None,
            "config": "full",
        },  # config -> small/imbalanced/plus/small_aug/full/intrinsic
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": "baseline",  # upsample/baseline
        "lr": 4e-5,
        "batch_size": 32,
        "epochs": 6,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy"]],
        "metric_best": "accuracy",
        "ngpu": 1,
        "gpt3_temp": 1.0,
        "eval_accumulation_steps": 30,
    }
)

snips_ex2_setup = hu.cartesian_exp_group(
    {
        # do multiple runs to account for stochasticity in metrics
        "run#": list(range(10)),
        "dataset": [
            {
                "name": "snips_official",
                "num_labels": 7,
                "oos_id": None,
                "config": "full_" + v,
            }
            for v in SNIPS_DOMAINS
        ],  # config -> small/imbalanced/plus/small_aug/full
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "exp_type": ["gpt3"],  # gpt3/upsample/baseline
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # metrics to compute. if oos_id is not None,
        # compute inscope_accuracy and oos_recall as well
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",
        # 'gpt3_engine': 'davinci',  # ada/babbage/curie/davinci/gptj
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # 0.5/0.6/0.7/0.8/0.9/1.0/1.5
        "eval_accumulation_steps": 30,
    }
)


fewshot_baseline_snips = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "snips_official",
            "num_labels": 7,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 3e-5,
        "batch_size": 8,
        "epochs": 20,
        "warmup_ratio": 0.1,
        "weight_decay": 0.001,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "baseline",
        "eval_accumulation_steps": 30,
    }
)

fewshot_oracle_snips = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "snips_official",
            "num_labels": 7,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3_oracle",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_gpt3_snips = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "snips_official",
            "num_labels": 7,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": "gpt3",
        "gpt3_engine": [
            "ada",
            "babbage",
            "curie",
            "davinci",
            "gptj",
        ],  # ada/babbage/curie/davinci/gptj
        # "gpt3_engine": "babbage",  # ada/babbage/curie/davinci/gptj
        "gpt3_temp": 1.0,  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "eval_accumulation_steps": 30,
    }
)

fewshot_eda_snips = hu.cartesian_exp_group(
    {
        "run#": list(range(10)),  # for extrinsic evaluation
        "dataset": {
            "name": "snips_official",
            "num_labels": 7,
            "oos_id": None,
            "config": "few_pure",
        },
        "model": {"name": "intent_classification", "backbone": "bert-large-uncased"},
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 10,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "metrics": [["accuracy", "f1", "precision", "recall"]],
        "metric_best": "accuracy",  # accuracy/f1
        "exp_type": ["eda", "eda_oracle"],
        "eval_accumulation_steps": 30,
    }
)

# CLINC
EXP_GROUPS["baselines"] = baselines
EXP_GROUPS["ex2_setup"] = ex2_setup
EXP_GROUPS["fewshot_baseline_clinc"] = fewshot_baseline_clinc
EXP_GROUPS["fewshot_oracle_clinc"] = fewshot_oracle_clinc
EXP_GROUPS["fewshot_gpt3_clinc"] = fewshot_gpt3_clinc
EXP_GROUPS["fewshot_gpt3mix_clinc"] = fewshot_gpt3mix_clinc
EXP_GROUPS["fewshot_eda_clinc"] = fewshot_eda_clinc

# Banking77
EXP_GROUPS["banking77_baselines"] = banking77_baselines
EXP_GROUPS["fewshot_baseline_banking77"] = fewshot_baseline_banking77
EXP_GROUPS["fewshot_gpt3_banking77"] = fewshot_gpt3_banking77
EXP_GROUPS["fewshot_oracle_banking77"] = fewshot_oracle_banking77
EXP_GROUPS["fewshot_eda_banking77"] = fewshot_eda_banking77

# HWU64
EXP_GROUPS["hwu64_baselines"] = hwu64_baselines
EXP_GROUPS["fewshot_baseline_hwu64"] = fewshot_baseline_hwu64
EXP_GROUPS["fewshot_gpt3_hwu64"] = fewshot_gpt3_hwu64
EXP_GROUPS["fewshot_oracle_hwu64"] = fewshot_oracle_hwu64
EXP_GROUPS["fewshot_eda_hwu64"] = fewshot_eda_hwu64

# SNIPS
EXP_GROUPS["snips_baselines"] = snips_baselines
EXP_GROUPS["snips_ex2_setup"] = snips_ex2_setup
EXP_GROUPS["fewshot_baseline_snips"] = fewshot_baseline_snips
EXP_GROUPS["fewshot_gpt3_snips"] = fewshot_gpt3_snips
EXP_GROUPS["fewshot_oracle_snips"] = fewshot_oracle_snips
EXP_GROUPS["fewshot_eda_snips"] = fewshot_eda_snips
