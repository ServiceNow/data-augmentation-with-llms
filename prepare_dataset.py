"""Dataset preparation step. It's mostly a one-time script."""
import os

os.environ["LOCAL_RANK"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/home/"
import argparse
from utils import main_data_utils, sample_few_shot, augment_slices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--name", default="clinc_oos")
    parser.add_argument(
        "--modes", nargs="+", default=["upsample", "gptj", "gpt3", "eda"]
    )
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.top_k < 0 or args.top_p > 1.0:
        print("args.top_k >= 0 | 0.0 <= args.top_p <= 1.0")
        import sys

        sys.exit()
    ds_config = main_data_utils.get_ds_config(args.name)
    print(f"Loaded dataset config for {args.name}")
    sample_few_shot(args.data_root, ds_config)
    print(f"augmenting for modes: {args.modes}")
    data_slices = augment_slices(
        args.data_root,
        ds_config,
        modes=args.modes,
        top_k=args.top_k,
        top_p=args.top_p,
    )
