# Audit of the Llama measurements

A second look at the language-model measurements of act III, after the film was finished: how much do its numbers depend on the automatic correctness labels, on chance, on the choice of block, on the wording of the questions, and on comparing two prompts? Everything here is recomputed by [`audit.py`](audit.py).

**In short: no statement of the film changes.** Two nuances deserve to be stated: the result on the answers that "look sure" rests on few wrong answers, and the best block was chosen among six.

## 1. The correctness labels

A response counts as right when, after normalisation, a TriviaQA alias appears in it (or it appears in an alias of more than three characters). Every E3 answer marked wrong (162), every flagged answer marked right, and the E5 answers that differ from E3 were read by hand ([`results/label_review.csv`](results/label_review.csv)).

- E3: 14 clear errors out of 600. Twelve right answers were marked wrong because the matching is literal ("Every 10 years" against "every ten years", "Psalm 117" against "Psalms 117", "Sigurd" for Siegfried). Two wrong answers were marked right ("demisemiquaver" contains "quaver"; a junk alias listing every M\*A\*S\*H character). Five more are debatable.
- E5: 13 clear errors (twelve and one), five debatable.

| | accuracy | probe | entropy | stated confidence | probe, answers that say 100 |
|---|---|---|---|---|---|
| automatic labels (the film) | 0.730 / 0.711 | 0.836 | 0.853 | 0.590 | 0.795 |
| reviewed labels | 0.747 / 0.730 | 0.853 | 0.862 | 0.589 | 0.802 |

Accuracy is given for E3 / E5. The review raises accuracy by about two points and the AUCs by about 0.01; it changes none of the comparisons. With the probe refitted on the reviewed labels, its AUC is 0.862.

## 2. Chance

95 % bootstrap intervals, 2,000 resamples, reviewed labels:

| | value | interval |
|---|---|---|
| E5: stated confidence | 0.589 | 0.540 to 0.637 |
| E5: probe | 0.816 | 0.776 to 0.854 |
| E5: probe minus stated confidence | 0.228 | 0.171 to 0.281 |
| answers that say 100: probe | 0.802 | 0.749 to 0.853 |
| answers that say 100: accuracy | 0.767 | 0.725 to 0.805 |
| E3: probe minus first-token entropy | -0.009 | -0.043 to 0.025 |
| output looks sure (n = 255): probe | 0.782 | 0.690 to 0.871 |
| output looks sure: entropy | 0.597 | 0.434 to 0.748 |
| output looks sure: probe minus entropy | 0.185 | 0.008 to 0.369 |
| E2: expected calibration error | 0.072 | 0.054 to 0.106 |

- The central result of III.5 is solid: the inside separates right from wrong answers far better than what the model says.
- Overall, the probe and the shape of the output are equivalent, as the film says.
- Where the output looks sure, the probe is clearly above chance and entropy is not. But that subset holds only 15 wrong answers (12 after review), and the advantage of the probe, while positive, is imprecise.

## 3. The choice of block

The film shows all six blocks and cites the best one as such. Choosing the block inside each training fold (nested cross-validation) gives 0.820 instead of 0.836. With other fold seeds, block 28 scores between 0.826 and 0.855.

## 4. The wording of the questions

Could the probe be reading the questions rather than the model? A probe on the words of the question alone scores 0.54, and the question length 0.53. The activations carry far more than the wording.

## 5. Two prompts

The probe reads the activations under the neutral prompt (E3); the stated confidence comes from a second prompt (E5) that also asks for a number. The two prompts give the same correctness on 91 % of the items. Scored against the E5 answers, the probe still reaches 0.81 (0.83 against its own answers). Note also that the probe is trained on the correctness of other questions, while the stated confidence is given without any training: the comparison measures what can be read inside the model, not a contest between equals.

## 6. Scope

One model (Llama-3.1-8B-Instruct), one dataset (TriviaQA), 600 questions, greedy decoding. Eleven E5 answers gave no readable confidence and are left out; nine of them were wrong under E3.
