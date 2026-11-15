# External data (not committed to git)

## synthetic_en/  organisers' synthetic English training set (SemEval-2027 Task 3)

Download `sentences.zip` and `metadata.zip` from the Codabench competition
(https://www.codabench.org/competitions/17510/) into `data/external/synthetic_en/`
and unzip both there, so the folder looks like this:

    data/external/synthetic_en/
        sentences.zip
        metadata.zip
        generated_sentences/<N>_senses/<lemma>.jsonl   2,345 words, 672,420 sentences
        metadata/<N>_senses/<lemma>.json               per sense: first-attestation year,
                                                       frequency per decade, tier

Each line of a .jsonl file is one sentence:
    {"lemma", "sense_id", "start", "end", "pos", "year", "text"}
`<N>_senses` = number of senses of the word; `year` = decade (1810-1990).

Windows note: two words differ only in case (Jock / jock), which a case-insensitive
file system cannot hold side by side. Extract the capitalised one as `Jock__cap.jsonl`
(the `lemma` field inside the file is what the code uses, not the file name).

About 2.6% of lines have a target span that is not the lemma (e.g. a synonym);
the training code filters them out.

Used ONLY to train the usage encoder of the proposed system. Never used for evaluation.
