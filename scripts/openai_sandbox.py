from utils.data_utils.augment_slices import openai_complete
import argparse


def main(args):
    if args.examples is None:
        raise ValueError("No seed examples provided.")
    if args.intent_name is None:
        raise ValueError("Please provide the name of a seed intent")
    if args.gpt_engine is None:
        print("No engine provided. Using ada...")
        ENGINE = "ada"
    else:
        ENGINE = args.gpt_engine

    intent = args.intent_name
    lines = args.examples
    k = len(args.examples)

    print("----Default method----")
    prompt = f"The following sentences belong to the same category {intent}:\n"
    prompt += "\n".join([f"Example {i+1}: {l}" for i, l in enumerate(lines)])
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
    parser.add_argument(
        "-i", "--intent_name", default=None, type=str, help="Name of the seed intent"
    )

    parser.add_argument(
        "-g",
        "--gpt_engine",
        default=None,
        help="GPT engine to use for augmentation (ada/babbage/curie/davinci)",
    )
    args, unknown = parser.parse_known_args()

    main(args)
