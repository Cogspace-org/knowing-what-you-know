# Measurements

Every number the film measured itself comes from one Colab notebook and one offline script:

- [`notebook/knowing_what_you_know_llama8b.ipynb`](notebook/knowing_what_you_know_llama8b.ipynb) runs the models (Llama-3.1-8B-Instruct; Qwen3-8B for effort). Greedy decoding throughout, so the outputs are deterministic: two complete runs gave identical files.
- [`derive.py`](derive.py) computes, from the notebook outputs, every number on screen that the notebook does not print directly, and checks it against the files the film used.

## Reproduce

```
python derive.py
```

Recomputes the derived numbers from `results/` and compares them with the shipped files. Only numpy is needed. Everything matches exactly, except two showcase scores rounded to three decimals and the full next-word distributions, stored in float16 (differences below 0.001).

```
python derive.py --activations activations_llama8b_layers.npz
```

Also refits the six probes from the activations (`activations_llama8b_layers.npz`, 84 MB, attached to the repository's release). About two minutes on a laptop; the AUCs and the out-of-sample scores come out identical.

To regenerate the outputs themselves, run the notebook on Colab with an A100 (about an hour). Llama 3.1 is gated: accept its licence on Hugging Face first.

## E1. The cold open

Four questions in plain chat mode, no instruction: Paris, The Garden of Earthly Delights, Alpha-three, and the capital of Chile. Files: `results/e1_*.json` (answer, top 50 at every step), `results/e1_*_step0.npy` (full first-step distribution).

| On screen | Value | Where |
|---|---|---|
| Paris, the gauge | 0.997 | `e1_paris.json`, `steps[0].chosen_prob` |
| Bosch, the gauge | 0.908 | `e1_bosch.json`, `steps[0].chosen_prob` |
| Alpha-three, the answer and the gauge | refusal, 0.49 ("I" and "There" tie) | `e1_alpha3.json`, `answer`, `steps[0].top` |

## E1b. The bars

The next-word distribution after an unfinished sentence, in completion mode (no chat template). Files: `results/e1b_*.json` (top 200, greedy continuation), `results/e1b_*_next.npy` (full distribution), `results/e1b_*_full.json` (the 3,000 largest probabilities, for the bars; recomputed by `derive.py`).

| On screen | Value | Where |
|---|---|---|
| "The capital of Chile is" | Santiago 0.692, entropy 2.68 bits | `e1b_santiago.json` |
| "The capital of the Martian colony Alpha-three is" | largest 0.199, 81 tokens above 0.1 %, entropy 7.27 bits | `e1b_alpha3.json` |
| New Eden | "a sprawling metropolis known as New Eden." | `e1b_alpha3.json`, `continuation` |
| vocabulary | 128,256 tokens | `e1b_*_full.json`, `n_vocab` |

## E2. Calibration

600 TriviaQA questions (validation, `rc.nocontext`, fixed seed), prompt `Answer the question with a short phrase only.` Confidence is the largest probability of the first answer token; correctness is checked against the TriviaQA aliases. Per item: `results/e3_items.jsonl`. Derived: `results/e2_reliability_before.json`.

| On screen | Value |
|---|---|
| accuracy | 0.73 |
| mean confidence | 0.688 |
| expected calibration error (ten bins) | 0.062 |
| top bin (n = 209) | confidence 0.969, accuracy 0.943 |

## E3 and H3. The probe

Same 600 questions. One forward pass per question gives the activations of blocks 8, 12, 16, 20, 24 and 28 at the last prompt token, before any word is generated. The probe is an L2-regularised logistic regression on standardised activations, scored out of sample with five stratified folds; it is fitted on TriviaQA only. Files: `results/h3_layer_scan.json`, `results/h3_metrics.json`, `results/h3_cloud.json` (the 3D cloud and the out-of-sample score of every question), `results/e3_showcase.json`, and derived `results/e3_numbers.json`.

| On screen | Value |
|---|---|
| the ladder, blocks 8 to 28 | 0.714, 0.739, 0.805, 0.809, 0.804, 0.836 |
| the shape of the output (first-token entropy) | 0.853 (maximum probability: 0.825) |
| when the output looks sure (entropy below 1 bit, n = 255) | probe 0.781, entropy 0.599 |
| the gauge | four right answers with the highest scores, four wrong ones with the lowest |

The 3D cloud is a PCA of the block-28 activations (6 % of the variance in three dimensions); its vertical axis is the out-of-sample probe score.

## E4. Effort

Qwen3-8B in thinking mode, greedy, three problems, a cap of 6,000 new tokens. File: `results/e4_effort.json`.

| Problem | Reasoning tokens |
|---|---|
| 17 + 26 | 359 |
| the two trains | 1,811 |
| n² + 2n + 12 a perfect square | 5,760, cut by the cap before the final answer |

## E5. What it says

The same 600 questions, prompt: answer with a short phrase, then `Confidence: N` from 0 to 100. Files: `results/e5_items.json` (everything, per question), `results/e5_said_vs_probe.json`; derived: `results/e5_analysis.json`, `results/e5_showcase_all.json`, `results/e5_showcase.json`. The probe is the block-28 out-of-sample score from E3, scored against the E5 answers.

| On screen | Value |
|---|---|
| answers with a readable confidence | 589 of 600 |
| says 100 | 400 times, right 75 % of those |
| average stated confidence | 95.3; 96.8 when right, 91.5 when wrong |
| accuracy | 0.711 |
| AUC of the stated confidence | 0.590 |
| AUC of the probe, same answers | 0.809; 0.795 among the answers that say 100 |
| unanswerable questions, neutral prompt (E3) | 24 of 30 refused or denied, 6 answers invented (`e3_unanswerable_labels.csv`, read by hand) |
| unanswerable questions, with a confidence (E5) | 7 of 30 answers invented, 6 with a confidence of 80 or more (`e5_unanswerable_labels.csv`, read by hand) |

## Files

| File | From |
|---|---|
| `results/e1_*`, `e1b_*.json`, `e1b_*.npy`, `e3_showcase.json`, `e4_effort.json`, `e5_items.json`, `e5_said_vs_probe.json`, `h3_*.json` | the notebook |
| `results/e3_items.jsonl` | the notebook's `activations_llama8b_layers.npz` without the activation vectors, plus the probe score from `h3_cloud.json` |
| `results/e2_reliability_before.json`, `e3_numbers.json`, `e5_analysis.json`, `e5_showcase_all.json`, `e1b_*_full.json` | `derive.py` |
| `results/e5_showcase.json` | four of the eight in `e5_showcase_all.json`, chosen for the screen |
| `results/e3_unanswerable_labels.csv`, `results/e5_unanswerable_labels.csv` | read by hand |

`e5_analysis.json` also stores the stated confidences on the 30 unanswerable questions (`unanswerable_said`). A non-zero confidence there does not mean an invented answer: several refusals come with "Confidence: 100" ("There is no Pan-Antarctic railway."). The labels file is the one to use.

## Licences of the material

The questions come from TriviaQA (Apache 2.0). The answers and activations are outputs of Llama-3.1-8B-Instruct (Llama 3.1 Community License) and Qwen3-8B (Apache 2.0).
