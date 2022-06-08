from utils.data_utils.augment_slices import openai_complete
import argparse

K = 10
ENGINE = "curie"
DATASET_NAME = "clinc_oos"
INTENT_1, INTENT_2 = "user_name", "timer"


def main(args):
    if args.examples is None:
        return ValueError("No seed examples provided.")
    lines1 = args.examples
    k = len(args.examples)

    print("----Default method----")
    prompt = f"The following sentences belong to the same category {INTENT_1}:\n"
    prompt += "\n".join([f"Example {i+1}: {l}" for i, l in enumerate(lines1)])
    prompt += "\n"
    prompt += f"Example {k+1}:"
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--examples",
        nargs="+",
        default=None,
        help="Seed examples for prompting GPT",
    )
    args, unknown = parser.parse_known_args()

    main(args)
