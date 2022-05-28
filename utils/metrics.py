import json, re, os, numpy as np

from datasets import load_metric
from utils import main_data_utils as mdu
from sklearn.metrics import confusion_matrix


class Metrics:
    def __init__(self, exp_dict=None, tokenizer=None):
        self.exp_dict = exp_dict
        self.tokenizer = tokenizer

    def compute_metrics(self):
        """
        Will choose the appropriate metric computer based on the config
        """
        if "bert" in self.exp_dict["model"]["backbone"]:
            return self.compute_metrics_bert
        raise ValueError(f"Incompatible backbone {self.exp_dict['model']['backbone']}.")

    def compute_metrics_bert(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {}
        # sort, so accuracy is evaluated first, always
        for metric in sorted(self.exp_dict["metrics"]):
            _metric = eval(f"self.{metric}")
            metrics.update(_metric(predictions, labels))
        return metrics

    def accuracy(self, predictions, labels):
        accuracies = {}
        acc = load_metric("accuracy")
        accuracies.update(acc.compute(predictions=predictions, references=labels))
        oos_id = self.exp_dict["dataset"]["oos_id"]
        if oos_id is not None:
            # compute in_scope accuracy as well
            inscope_preds, inscope_labels = [], []
            for idx in range(len(labels)):
                if labels[idx] == oos_id:
                    continue
                inscope_labels.append(labels[idx])
                inscope_preds.append(predictions[idx])
            self.inscope_preds, self.inscope_labels = inscope_preds, inscope_labels
            accuracies["inscope_accuracy"] = acc.compute(
                predictions=inscope_preds, references=inscope_labels
            )["accuracy"]
        return accuracies

    def f1(self, predictions, labels):
        f1s = {}
        f1 = load_metric("f1")
        f1s.update(
            f1.compute(predictions=predictions, references=labels, average="macro")
        )
        if self.exp_dict["dataset"]["oos_id"] is not None:
            f1s["inscope_f1"] = f1.compute(
                predictions=self.inscope_preds,
                references=self.inscope_labels,
                average="macro",
            )["f1"]
        return f1s

    def precision(self, predictions, labels):
        precision = load_metric("precision")
        return precision.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def recall(self, predictions, labels):
        recalls = {}
        recall = load_metric("recall")
        recalls.update(
            recall.compute(predictions=predictions, references=labels, average="macro")
        )
        oos_id = self.exp_dict["dataset"]["oos_id"]
        if oos_id is not None and oos_id in labels:
            # compute OOS recall
            outscope_preds = []
            for idx in range(len(labels)):
                if labels[idx] == oos_id:
                    outscope_preds.append(1 if predictions[idx] == oos_id else -1)
            recalls["oos_recall"] = outscope_preds.count(1) / len(outscope_preds)
        return recalls

    def confusion_matrix(self, predictions, references):
        return {
            "confusion_matrix": confusion_matrix(
                y_true=references,
                y_pred=predictions,
                labels=list(range(self.exp_dict["dataset"]["num_labels"])),
            )
        }

    def compute_fidelities(self, ds):
        """
        Returns and saves fidelities for all engines for given a barebone exp_dict
        containing dataset name, num_labels, few_pure setting
        its number of classes
        Example schema of the returned result
        eda: Float
        ada_1.0: Float
        .
        .
        .
        gptj_2.0: Float
        NOTE: al_dataset.pkl must've been generated already for the dataset
        """
        al_ds_path = mdu.pjoin("data", ds, "full", "al_dataset.pkl")
        if not os.path.exists(al_ds_path):
            print("Oracle relabelling hasn't been done on the generated samples yet")
            print("Go run runners.oracle_relabel first")
            return
        generated_ds = mdu.read_pickle(al_ds_path)["generated"]
        # path to save this dataset's fidelity results
        results_path = f"results/{ds}_fidelity.json"
        oos_id = mdu.read_json(f"data/{ds}/name2id.json").get("oos")
        if os.path.exists(results_path):
            print(f"Loading {results_path} that already exists!")
            print(f"Delete/Rename it to compute fidelity for {ds} again")
            return mdu.read_json(results_path)

        print(f"Computing fidelity numbers for {ds}")
        results = {}
        for engine, samples in generated_ds.items():
            if engine == "val":
                engine = "threshold"
            _a = [
                1 if old == new else 0
                for old, new in zip(samples["old_intent"], samples["intent"])
            ]
            results[engine] = np.mean(_a)

        print(f"Saving fidelity numbers for {ds}")
        mdu.write_json(results, results_path)
        return results
