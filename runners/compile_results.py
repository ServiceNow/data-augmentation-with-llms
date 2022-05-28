import os, json, copy, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from utils.metrics import Metrics

# this dir contains CLINC and SNIPS partial fewshot experiments
# EXPERIMENTS_DIR = "/path/to/partial-fewshot/savedir/"

# this dir contains CLINC, SNIPS, Banking77, and HWU64 full fewshot experiments
EXPERIMENTS_DIR = "/path/to/full-fewshot/savedir/"

# FILTERS (NOTE: any of these filters can be disabled by setting to None)
datasets = None  # clinc_oos, snips_official, banking77, hwu64
augmentation = None  # eda (in FS), ada, babbage, curie, davinci, gptj
temperatures = None  # 0.5-2.0
skip_oracle = False  # True/False

# exp_type-wise significance tests are always computed w.r.t davinci models
TTEST = False  # True/False

# print results as a latex table
TO_LATEX = False

# save metrics for plotting purposes
GEN_FOR_PLOT = False
# name of the results file (NOTE: only GPTJ for full fewshot is supported)
FNAME = "results/gptj_val_fidelity_et_al.pkl"

# The below vars are used by to_latex()
# ALL_METRICS and ALL_DATASETS are in the order or reporting in the paper
# defining the order here so that we don't need to apply any convoluted
# ops like sorting to ensure consistent performance reporting.
# NOTE: to_latex() auto-ignores a dataset if it's filtered out in this script
ALL_DATASETS = ["clinc_oos", "hwu64", "banking77", "snips_official"]
BACKBONE = "BERT" if "bert" in EXPERIMENTS_DIR else "T5"
SCOPES = ["overall", "few_shot"] if "ex2" in EXPERIMENTS_DIR else ["test"]
ALL_METRICS = ["IA", "OR"]

# No need to touch anything else below this line.
FILTER = {
    "datasets": datasets,  # set None to fetch for all datasets
    "augmentation": augmentation,  # set None to fetch for all augmentors
    "temps": temperatures,  # set None to fetch for all temperatures
    "skip_oracle?": skip_oracle,  # set False/None to fetch oracle results
}

pjoin = os.path.join


def fmt(xs):
    return f"{np.mean(xs):.2f} ({np.std(xs):.2f})"


def compile_results(folder_list, exp_dir):
    # fetch experiment type, and populate total_results accordingly
    exp_dict_path = pjoin(exp_dir, folder_list[0], "exp_dict.json")
    exp_dict = json.load(open(exp_dict_path, "r"))
    dataset = exp_dict["dataset"]["name"]
    dconfig = exp_dict["dataset"]["config"]
    oos_id = exp_dict["dataset"]["oos_id"]
    exp_type = exp_dict["exp_type"]
    ex2_setup = True if dconfig.startswith("full_") else False
    full_fewshot = True if dconfig == "few_pure" else False

    if dconfig != "few_pure" and not ex2_setup and exp_type == "baseline":
        org_baseline = True
    else:
        org_baseline = False

    # declare total_results template based on exp config
    if full_fewshot or org_baseline:  # no need for overall, fewshot keys
        total_results = {
            "test": {"IA": [], "OR": [], "A": []},
            "val": {"IA": [], "OR": [], "A": []},
        }
        if oos_id is None:
            total_results = {
                "test": {"A": []},
                "val": {"A": []},
            }
    else:  # ex2 setup
        total_results = {
            "few_shot": {"IA": []},
            "few_shot_val": {"IA": []},
            "overall": {"IA": [], "OR": []},
            "overall_val": {"IA": [], "OR": []},
        }

    # read all the json files
    for folder in folder_list:
        sub_results = json.load(open(pjoin(exp_dir, folder, "code/results.json")))
        key = list(sub_results.keys())[0]
        sub_results = sub_results[key]

        if dataset == "snips_official" and ex2_setup:  # snips has no OR
            _overall = sub_results["overall"]
            total_results["overall"]["IA"].append(_overall["test_accuracy"])
            total_results["overall_val"]["IA"].append(_overall["valid_accuracy"])

        elif full_fewshot or org_baseline:
            total_results["test"]["A"].append(sub_results["test_accuracy"])
            total_results["val"]["A"].append(sub_results["valid_accuracy"])
            if not oos_id:
                continue
            total_results["test"]["IA"].append(sub_results["test_inscope_accuracy"])
            total_results["val"]["IA"].append(sub_results["valid_inscope_accuracy"])
            total_results["test"]["OR"].append(sub_results["test_oos_recall"])
            total_results["val"]["OR"].append(sub_results["valid_oos_recall"])
            continue

        elif ex2_setup and dataset == "clinc_oos":
            # not handling oos_id as ex2_setup ALWAYS has an oos_id
            overall = sub_results["overall"]
            total_results["overall_val"]["IA"].append(overall["valid_inscope_accuracy"])
            total_results["overall"]["IA"].append(overall["test_inscope_accuracy"])
            total_results["overall_val"]["OR"].append(overall["valid_oos_recall"])
            total_results["overall"]["OR"].append(overall["test_oos_recall"])

        fs = sub_results["few_shot"]
        total_results["few_shot"]["IA"].append(fs["test_accuracy"])
        total_results["few_shot_val"]["IA"].append(fs["valid_accuracy"])

    for k in total_results:
        for m in total_results[k]:
            results = [100 * v for v in total_results[k][m]]
            # the following is averging across different fewshot domains.
            # For EX2, it's nth run's avg. across full_banking, full_meta, etc.
            # For Full fewshot, it's computing the mean for one domain, i.e,
            # the value is going to remain the same except it won't be a list.
            total_results[k][m] = np.mean(results)
    return total_results


def segregate_sub_folders(exp_dir):
    sub_folder_dict = {}

    for folder in os.listdir(exp_dir):
        exp_dict_path = pjoin(exp_dir, folder, "exp_dict.json")
        exp_dict = json.load(open(exp_dict_path))
        dname = exp_dict["dataset"]["name"]  # aggregate on the dataset

        # dataset filter
        if FILTER["datasets"] and dname not in FILTER["datasets"]:
            continue
        exp_dict["gpt3_temp"] = 1.0  # TODO: remove
        if "gpt" in exp_dict["exp_type"]:
            engine, temp = exp_dict["gpt3_engine"], exp_dict["gpt3_temp"]
            # engine and temperature filter
            if FILTER["augmentation"] and engine not in FILTER["augmentation"]:
                continue
            if FILTER["temps"] and temp not in FILTER["temps"]:
                continue
            exp_type = f"{exp_dict['exp_type']}_{engine}_{temp}"
        else:
            # NOTE that for non-GPT experiments exp_type is the augmentation mode
            exp_type = exp_dict["exp_type"]  # eda/eda_oracle
            _aug = exp_type.replace("_oracle", "") if "oracle" in exp_type else exp_type
            if FILTER["augmentation"] and _aug not in FILTER["augmentation"]:
                continue

        # oracle filter
        if "oracle" in exp_type and FILTER["skip_oracle?"]:
            continue

        if exp_type not in sub_folder_dict:
            sub_folder_dict[exp_type] = {}
        if dname not in sub_folder_dict[exp_type]:
            sub_folder_dict[exp_type][dname] = defaultdict(list)
        sub_folder_dict[exp_type][dname][exp_dict["run#"]].append(folder)

    # a sanity check line, prints the number of experiments per config.
    folders_per_exp = []
    for exp_type in sub_folder_dict:
        for dname in sub_folder_dict[exp_type]:
            num_runs = len(sub_folder_dict[exp_type][dname])
            folders_per_exp.append((exp_type, dname, num_runs))

    print(folders_per_exp, len(folders_per_exp))
    return sub_folder_dict


def final_compile(sub_results_dicts):
    final_result = {}
    for s in sub_results_dicts:
        for scope in s.keys():
            if scope not in final_result:
                final_result[scope] = {}
            for metric in s[scope]:
                if np.isnan(s[scope][metric]):
                    continue
                if metric not in final_result[scope]:
                    final_result[scope][metric] = []
                final_result[scope][metric].append(s[scope][metric])
    return final_result


def get_performance(exp_dir):
    """
    returns a results dictionary which is not aggregated by runs

    An example of hierarchy:
    gpt3_ada_1.0:
        clinc_oos:
            test:
                IA[90.32, 90.1...90.3]
                OR[40.23, 39.12...38.1]
            val:
                IA[92.32, 91.1...93.3]
                OR[45.23, 41.12...40.1]
        banking77:
            test:
                A[82.3...80.1]
            val:
                A[83.2...79.2]
        snips_official:
            .
            .
            .
    gpt3_babbage_1.0:
        .
        .
        .

    It's not aggregated so that other functions may use to for:
        - mean and std computation across multiple runs
        - significace testing
    """
    sub_folder_dict = segregate_sub_folders(exp_dir)
    performance = {}
    for exp_type in sorted(list(sub_folder_dict.keys())):
        for dname in sorted(list(sub_folder_dict[exp_type].keys())):
            config_results = []
            for config in sub_folder_dict[exp_type][dname]:
                folderlist = sub_folder_dict[exp_type][dname][config]
                config_results.append(compile_results(folderlist, exp_dir))
            if exp_type not in performance:
                performance[exp_type] = {}
            performance[exp_type][dname] = final_compile(config_results)
    return performance


def to_latex(performance):
    """
    Generates latex table code for aug, aug+relabel settings
    """
    table_latex = ""
    # backbone mode (aug/aug.+relabel)
    template = "{} {} (Ours) &"  # line template

    # num of columns to report will be same for all exp. settings
    _etype = list(performance.keys())[0]
    n_cols = 0
    for dname in performance[_etype]:
        for s in SCOPES:
            curr_metrics = performance[_etype][dname][s]
            for _m in curr_metrics:
                if _m == "A" and "IA" in curr_metrics:
                    continue
                n_cols += 1

    template += " {} &" * (n_cols - 1)
    template += " {} \\\\\n"

    for etype in performance:
        dscores = []
        for dname in ALL_DATASETS:
            # print(dname)
            if dname not in performance[etype]:
                continue
            # print(dname)
            for s in SCOPES:
                # print(s)
                curr_metrics = list(performance[etype][dname][s])
                for _m in ALL_METRICS:
                    if _m not in curr_metrics:
                        # a dataset without IA means no OOS. in that case,
                        # A is the same as IA.
                        if _m == "IA":
                            _m = "A"
                        else:
                            continue
                    dscores.append(fmt(performance[etype][dname][s][_m]))
                    # print(_m)
            # print("===")

        table_latex += template.format(
            BACKBONE, etype.replace("_1.0", "").replace("_", "\_"), *dscores
        )
    print(table_latex)


def perform_ttest(performance):
    """
    receives a performance for datasets and performs two statistical
    t-tests w.r.t davinci model at 1.0 temp. for that experiment type
    """
    if performance == {}:
        print("Nothing to show here")
        return

    bigger_model = None
    for e in performance:
        if "davinci" in e:
            bigger_model = e
    bigger_results = performance[bigger_model]

    # Gather model-wise metrics
    for dname in bigger_results.keys():
        print(f"Dataset: {dname.upper()}")
        print("-" * 30)
        for s in SCOPES:
            for m in bigger_results[dname][s]:
                _bresults = bigger_results[dname][s][m]
                print(f"--- {s} {m} test ---")
                print(f"{bigger_model.upper()}: ({fmt(_bresults)})")
                for model, results in performance.items():
                    if model == bigger_model:
                        continue
                    _sresults = results[dname][s][m]
                    test_result = stats.ttest_ind(_bresults, _sresults)
                    print(f" vs {model.upper()} ({fmt(_sresults)}) {test_result}")
            print()


def display_results(performance):
    for etype in performance:
        for dname in performance[etype]:
            for scope in performance[etype][dname]:
                for metric in performance[etype][dname][scope]:
                    results = performance[etype][dname][scope][metric]
                    performance[etype][dname][scope][metric] = fmt(results)

    for etype in performance:
        for dname in performance[etype]:
            print(f"Setting: {etype} | {dname}")
            print("-" * 20)
            print(pd.DataFrame().from_dict(performance[etype][dname]))
            print("=" * 30)
            print("\n")


def gen_for_plot(performance):
    """
    Will save fidelity and fs accuries for all the datasets in a file
    NOTE: this doesn't save metrics for partial fewshot temp. profiling nor
    does it support any engine other than GPTJ right now.
    """
    if FILTER["augmentation"] != ["gptj"]:
        raise NotImplementedError(
            "Metrics generation for plotting only supported for GPTJ!"
        )
    if "fs" not in EXPERIMENTS_DIR:
        raise NotImplementedError(
            "Metrics generation for plotting only supported for Full Fewshot!"
        )

    if os.path.exists(FNAME):
        print(f"{FNAME} already exists!! Loading...")
        print("Delete/Rename it to recompute fidelities.")
        return pickle.load(open(FNAME, "rb"))
    print("Compiling plotting metrics in a file...")

    # init df
    df = pd.DataFrame(columns=["temp", "ds", "val_acc_mean", "val_acc_std", "fidelity"])

    # compute fidelities
    fidelities = {ds: Metrics().compute_fidelities(ds) for ds in ALL_DATASETS}
    for etype, results in performance.items():
        # etype: gpt3_gptj_1.0
        _, temp = etype.rsplit("_", 1)
        for ds in results:
            acc_key = "IA" if ds == "clinc_oos" else "A"
            val_accs = results[ds]["val"][acc_key]
            val_acc_mean, val_acc_std = np.mean(val_accs), np.std(val_accs)
            _fid = fidelities[ds][f"gptj_{temp}"]
            # create a new entry in the dataframe
            df.loc[len(df.index)] = [float(temp), ds, val_acc_mean, val_acc_std, _fid]

    # will be used to plot the threshold lines in the fidelity plots
    thresholds = {ds: fidelities[ds]["threshold"] for ds in ALL_DATASETS}
    metrics = {"metrics": df, "thresholds": thresholds}
    print(f"Saving fidelity metrics for plotting {FNAME}")
    with open(FNAME, "wb") as f:
        pickle.dump(metrics, f)
    return metrics


def main():
    """
    Computes mean and std of metrics obtained by get_performance
    """
    # remove the "deleted" folder if it exists
    if os.path.exists(pjoin(EXPERIMENTS_DIR, "deleted")):
        print(f"Removing {pjoin(EXPERIMENTS_DIR, 'deleted')}...")
        os.system(f"rm -rf {pjoin(EXPERIMENTS_DIR, 'deleted')}")
        print("Removed.")

    performance = get_performance(EXPERIMENTS_DIR)

    # display results (deepcopy needed as display_results permutes its input)
    display_results(copy.deepcopy(performance))

    if TTEST:
        # non-oracle etype
        print("T-Test no oracle")
        perform_ttest({k: v for k, v in performance.items() if "oracle" not in k})

        # oracle etype
        print("T-Test with oracle")
        perform_ttest({k: v for k, v in performance.items() if "oracle" in k})

    if TO_LATEX:
        to_latex(performance)

    if GEN_FOR_PLOT:
        gen_for_plot(performance)


if __name__ == "__main__":
    main()
