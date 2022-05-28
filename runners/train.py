from multiprocessing.sharedctypes import Value
from configs import exp_configs
import os, argparse, torch, wandb
from haven import haven_wizard as hw

from runners.modules import bert

torch.backends.cudnn.benchmark = True


def init_wandb(exp_dict):
    exp_name = f"{exp_dict['exp_type']}_oosID={exp_dict['dataset']['oos_id']}_"
    exp_name += f'{exp_dict["dataset"]["name"]}_{exp_dict["dataset"]["config"]}_'
    exp_name += f'{exp_dict["lr"]}_{exp_dict["epochs"]}_{exp_dict["batch_size"]}'
    wandb.init(project="few-shot-nlp", name=exp_name)


def trainval(exp_dict, savedir, args):
    """Main."""
    # ==========================
    # init wandb
    # ==========================
    if not args.disable_wandb:
        init_wandb(exp_dict)
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # ==========================
    # Load appropriate trainer
    # ==========================
    if "bert" in exp_dict["model"]["backbone"]:
        return bert.trainval(exp_dict, savedir, args)
    return ValueError("backend not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group_list",
        nargs="+",
        default="resnet",
        help="name of an experiment in exp_configs.py",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/mnt/home/haven_output",
        help="folder where logs will be saved",
    )
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-d", "--datadir", type=str, default="./data")
    parser.add_argument("-md", "--modeldir", type=str, default=None)
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Overwrite previous results"
    )
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument(
        "-j",
        "--job_scheduler",
        type=str,
        default=None,
        help="If 1, runs in toolkit in parallel",
    )
    parser.add_argument("-v", default="results.ipynb", help="orkestrator")
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )
    parser.add_argument("--disable_wandb", default=1, type=int)
    parser.add_argument("--retain_checkpoints", action="store_true")

    args, unknown = parser.parse_known_args()
    if args.job_scheduler == "1":
        from configs import job_configs

        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    file_name = os.path.basename(__file__)[:-3]  # remove .py
    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_config,
        python_binary_path=args.python_binary,
        python_file_path=f"-m runners.{file_name}",
        use_threads=True,
        args=args,
        results_fname="notebooks/fsn.ipynb",
    )
