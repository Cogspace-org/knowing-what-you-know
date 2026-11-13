# Knowing What You Know

The written notes, the sources and the code behind *Knowing What You Know*, a documentary on metacognition from Socrates to neurons to machines, by COGSPACE.

| | |
|---|---|
| [NOTES.md](NOTES.md) | Chapter by chapter: what the film claims, where it comes from, what it simplifies. Corrections at the end. |
| [REFERENCES.md](REFERENCES.md) | Every source, with full references. |
| [measurements/](measurements/README.md) | Every number the film measured on a language model: the notebook that produced it, its outputs, and a script that recomputes and checks the rest. |
| [transcript/](transcript/transcript.md) | The narration, chapter by chapter, and the English subtitles. |

## Check the numbers

```
cd measurements
python derive.py
```

Needs Python 3 and numpy. The script recomputes every derived number from the notebook outputs and compares it with the files the film used. With the activations attached to the release, it also refits the probes:

```
python derive.py --activations activations_llama8b_layers.npz
```

## Corrections

If you see a mistake, open an issue. That is rather the point of the film.

## Licence

Code (the notebook and `derive.py`): MIT, see [LICENSE](LICENSE). Text (notes, transcript): CC BY 4.0. The questions come from TriviaQA (Apache 2.0); the answers and activations are outputs of Llama-3.1-8B-Instruct (Llama 3.1 Community License) and Qwen3-8B (Apache 2.0).
