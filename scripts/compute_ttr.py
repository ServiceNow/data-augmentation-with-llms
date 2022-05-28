import pickle
import nltk


def ttr(sentences, n):
    """
    Computes Type-Token Ratio for a given
    set of sentences
    Params:
    ======
    sentences (list): list of sentences in the corpus
    n (list): list of n values to consider when computing ngrams
    """
    ngrams = [[] for _ in n]
    for s in sentences:
        tokens = nltk.tokenize.word_tokenize(s.lower())
        for i in range(len(n)):
            ngrams[i].extend(list(nltk.ngrams(tokens, n[i])))
    return [round(len(set(v)) / len(v), 2) for v in ngrams]


def main():
    for dataset in ["clinc_oos", "snips_official", "hwu64", "banking77"]:
        dpath = f"./data/{dataset}/full/data_full_suite.pkl"
        data = pickle.load(open(dpath, "rb"))
        domains = list(data.keys())
        print(f"Diversity metrics for {dataset}")
        for e in ["eda", "ada", "babbage", "curie", "davinci"]:
            for t in [1.0]:
                sentences = []
                for d in domains:
                    # skip OOS for now
                    if d == "oos":
                        continue
                    attr = e if e == "eda" else f"{e}_{t}"
                    sentences.extend(data[d]["F"][attr]["text"])
                print(f"{e}_{t}: {ttr(sentences, n=[1, 2, 3, 4])}")


if __name__ == "__main__":
    main()
