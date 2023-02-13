from typing import Optional
import re
import spacy
import typer
from itertools import islice
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from datasets import load_dataset
from more_itertools import chunked

import spacy
from mecab import MeCab, Morpheme
from spacy import Vocab
from spacy.lang.ko.tag_map import TAG_MAP
from spacy.scorer import Scorer
from spacy.symbols import POS, X
from spacy.tokens import Doc
from spacy.training import validate_examples
from spacy.util import DummyTokenizer


class KoreanTokenizer(DummyTokenizer):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.mecab = MeCab()

    def __reduce__(self):
        return KoreanTokenizer, (self.vocab,)

    def __call__(self, text: str) -> Doc:
        dtokens = self.mecab.parse(text)
        surfaces = [dt.surface for dt in dtokens]

        doc = Doc(self.vocab, words=surfaces, spaces=list(check_spaces(text, surfaces)))

        for token, dtoken in zip(doc, dtokens):
            token.tag_ = dtoken.pos
            if token.tag_ in TAG_MAP:
                token.pos = TAG_MAP[token.tag_][POS]
            else:
                token.pos = X
            token.lemma_ = convert_expression(dtoken)
        doc.user_data["full_tags"] = [dt.pos for dt in dtokens]

        return doc

    def score(self, examples):
        validate_examples(examples, "KoreanTokenizer.score")
        return Scorer.score_tokenization(examples)


def convert_expression(m: Morpheme):
    expr = m.feature.expression

    if expr is None:
        return m.surface
    else:
        return "+".join([e.split("/")[0] for e in expr.split("+")])


def check_spaces(text, tokens):
    prev_end = -1
    start = 0
    for token in tokens:
        idx = text.find(token, start)
        if prev_end > 0:
            yield prev_end != idx
        prev_end = idx + len(token)
        start = prev_end
    if start > 0:
        yield False


@spacy.registry.tokenizers("mgylabs_korean_tokenizer")
def create_mgylabs_korean_tokenizer():
    def create_tokenizer(nlp):
        return KoreanTokenizer(nlp.vocab)

    return create_tokenizer


def tokenize_batch(nlp, batch):
    output = []
    texts = (re.sub(r"\s+", " ", line.strip()) for line in batch)
    for doc in nlp.pipe(texts):
        for sent in doc.sents:
            output.append(" ".join([t.text for t in sent]) + "\n")
    return output


def main(
    lang: str,
    output_file: Path,
    input_file: Optional[Path] = None,
    input_dataset: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    dataset_split: Optional[str] = None,
    dataset_streaming: bool = True,
    dataset_auth_token: bool = False,
    max_texts: int = -1,
    n_process: int = 8,
    batch_size: int = 1000,
):
    if input_file is None and input_dataset is None:
        raise ValueError("Provide either an input file or an input dataset.")

    if lang == "ko":
        nlp = spacy.blank(
            "ko", config={"nlp": {"tokenizer": {"@tokenizers": "mgylabs_korean_tokenizer"}}}
        )
    elif lang == "zh":
        nlp = spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "pkuseg"}}})
        nlp.tokenizer.initialize(pkuseg_model="spacy_ontonotes")
    else:
        nlp = spacy.blank(lang)

    nlp.add_pipe("sentencizer")
    nlp.max_length = 10**8

    if input_file:
        if max_texts > 0:
            texts = islice(open(input_file), max_texts)
        else:
            texts = open(input_file)
    elif input_dataset:
        dataset = load_dataset(
            input_dataset,
            dataset_subset,
            split=dataset_split,
            streaming=dataset_streaming,
            use_auth_token=dataset_auth_token,
        )
        if max_texts > 0:
            texts = (line["text"] for line in islice(iter(dataset), max_texts))
        else:
            texts = (line["text"] for line in dataset)

    with open(output_file, "w", encoding="utf-8") as output_fileh, Pool(processes=n_process) as pool:
        result = pool.imap(partial(tokenize_batch, nlp), chunked(texts, batch_size))
        for lines in result:
            output_fileh.writelines(lines)


if __name__ == "__main__":
    typer.run(main)
