# Emotion Steering in Open LLMs

A project where we tried to see if we can make open-source language models "feel" different emotions by poking at their internals. Inspired by Anthropic's April 2026 research, but done on free Kaggle GPUs with open models.

---

## What is this project about?

Anthropic found that their Claude model has internal "emotion vectors" — basically directions in its math-space that correspond to real emotions like happiness, fear, or desperation. If you push the model in that direction while it's generating text, its behavior actually changes.

We wanted to check: **does this also work on open-source models that anyone can download?**

We tested three models:
- `meta-llama/Llama-3.1-8B` (base)
- `nvidia/Mistral-NeMo-Minitron-8B-Base`
- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`

Short answer: **yes, it works.** All three models passed our tests.

---

## How does it work? (Simple version)

Language models are basically giant math functions. At each layer, they produce a big list of numbers (a vector) that represents what they "know" so far. 

We found that if you look at these vectors when the model is processing emotional text, vectors for the same emotion tend to point in the same direction in that math-space. 

So we:
1. Collected the vectors from lots of emotional stories
2. Averaged them to get one "emotion direction" per emotion
3. Added that direction to the model's brain while it was generating text
4. Watched the model's output change accordingly

---

## The 5 Emotions We Tested

| Emotion | Example implicit prompt used in testing |
|---|---|
| `desperate` | *"I just got the third rejection letter this week and rent is due tomorrow"* |
| `calm` | *"The lake was perfectly still at dawn, not a single ripple"* |
| `happy` | *"She opened the envelope and read that she'd been accepted into her top choice"* |
| `afraid` | *"The pilot's voice came over the intercom: 'We're experiencing some... difficulties'"* |
| `angry` | *"He found out his coworker had been taking credit for his work for six months"* |

---

## What we actually did (step by step)

### Step 0: Quick sanity check
First we just wanted to confirm that steering does *something*. We used a library called `repeng` to quickly extract a `desperate` vs `calm` vector and check if generated text changes. It did, so we moved on.

### Step 1: Generate 300 stories
We used GPT-4o-mini to write 300 short emotional stories — 60 per emotion, across 20 different everyday topics (work, family, health, finance, etc.).

The stories couldn't use the emotion word directly. They had to *show* the emotion through actions and sensations. This keeps the data cleaner.

We also pulled 100 neutral Wikipedia paragraphs to use as a baseline.

### Step 2: Extract the model's internal thoughts
We ran each story through the model and saved what was happening inside it at specific layers. Think of it like taking an X-ray at 10 different moments as the model reads the story.

For Llama 3.1 8B (32 layers total), we checked layers: 8, 10, 13, 15, 17, 19, 21, 24, 26, 29.

### Step 3: Calculate emotion vectors
For each layer, we subtracted the "average of everything" from the "average of each emotion" to get a direction vector unique to that emotion. Then we cleaned up noise using PCA and normalized everything.

We found the best layer by checking where positive emotions (happy, calm) and negative emotions (desperate, afraid, angry) point in the most opposite directions.

### Step 4: Test if it actually works
Three tests:
1. **Logit Lens** — Can we decode the emotion from the vector by looking at what words it predicts? (This failed for all models — interesting finding!)
2. **Implicit Scenario Probe** — If we show the model a situation that *implies* an emotion (without naming it), does the correct emotion vector score highest? (This passed!)
3. **Behavioral Steering** — If we inject the `desperate` vector while the model generates text, does it actually behave more desperately? (Yes! Tested with an impossible coding task)

---

## Results Per Model

### Meta Llama 3.1 8B
- Best layer: **19**
- Emotions `afraid` and `angry` were very similar internally (cosine similarity: 0.879)
- Positive and negative emotions separated cleanly in PCA
- Scenario Probe Score: **2/5** — PASS ✅

### NVIDIA Mistral-NeMo-Minitron 8B Base
- Best layer: **18**
- The strongest separation of all three models
- `afraid` and `angry` were the most similar of any pair we tested (0.885)
- Scenario Probe Score: **3/5** — PASS ✅ (best performer)

### NVIDIA Llama 3.1 Nemotron Nano 8B v1
- Best layer: **19**
- Emotions were more spread out across PCA space compared to the other two
- `desperate` and `angry` had the highest similarity (0.856)
- Scenario Probe Score: **2/5** — PASS ✅

### Summary Table

| Model | Best Layer | Probe Score | Result |
|---|---|---|---|
| Llama 3.1 8B | 19 | 2/5 | ✅ PASS |
| Mistral-NeMo-Minitron 8B | 18 | 3/5 | ✅ PASS |
| Nemotron Nano 8B v1 | 19 | 2/5 | ✅ PASS |

---

## Interesting things we noticed

- **Negative emotions cluster together.** Across all models, `desperate`, `afraid`, and `angry` always had high cosine similarity (0.83–0.89). The model kind of groups "bad/threatening feelings" together internally.
- **Logit Lens failed every time.** We couldn't decode the emotion from the vector by looking at vocabulary predictions. This is weird and might mean emotions are encoded at a level below individual words. Or it could be a 4-bit quantization side effect.
- **Steering is causal, not just correlational.** When we added `+desperate` at different strengths (0.05 → 0.1), the effect scaled up. A stronger injection = stronger behavioral change. That monotonic relationship is what distinguishes a real cause from a fluke.

---

## Files in this repo

| File | What it does |
|---|---|
| `complete-steering-pipeline.ipynb` | The main end-to-end pipeline notebook |
| `complete-steering-pipeline-2.ipynb` | Improved version with better checkpointing |
| `llama-research.ipynb` | Llama 3.1 8B specific experiments |
| `llama-3-1-nemotron-nano-8b-v1.ipynb` | Nemotron Nano 8B experiments |
| `mistral-nemo-minitron-8b-base.ipynb` | Minitron 8B experiments |
| `minitron-steering.ipynb` | Extra steering analysis for Minitron |
| `STEERING RESULTS.pdf` | All the plots, heatmaps, and charts in one place |

---

## How to run this yourself

### What you need
- A free [Kaggle](https://kaggle.com) account (for the T4 GPU)
- A [HuggingFace](https://huggingface.co) account (for model access)
- Accept Meta's license for Llama 3.1 on the HuggingFace model page
- Add your HF token to Kaggle secrets as `HF_TOKEN`

### Install dependencies
```bash
pip install transformers bitsandbytes accelerate repeng
pip install numpy scikit-learn matplotlib seaborn tqdm datasets
pip install openai  # only needed for story generation in Phase 1
```

### Running
1. Open `complete-steering-pipeline.ipynb` on Kaggle
2. Run cells top to bottom
3. Checkpoints are saved after each phase so you won't lose progress if the session dies
4. If you want model-specific details, open the individual notebooks

---

## What's next (ideas)

- Test on **Nemotron-H 8B** which is a hybrid Mamba + Transformer model — do emotion vectors exist in Mamba layers too?
- Run without 4-bit quantization to see if Logit Lens works better in full precision
- Try more emotions (fear of failure, nostalgia, confusion, etc.)
- Test on bigger models (70B) to see if the effect gets stronger

---

## Tech stack

| Library | Used for |
|---|---|
| `transformers` | Loading models |
| `bitsandbytes` | 4-bit quantization so models fit on T4 |
| `repeng` | Quick smoke test in Phase 0 |
| `scikit-learn` | PCA, cosine similarity |
| `matplotlib` / `seaborn` | Plots and heatmaps |
| `openai` | Story generation |

---

*Built on free Kaggle GPUs. No model training — just looking at what's already inside.*
