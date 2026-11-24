# Open Desk: an AI support desk you can look inside

[![tests](https://github.com/Atharv1414/ai-support-desk/actions/workflows/ci.yml/badge.svg)](https://github.com/Atharv1414/ai-support-desk/actions/workflows/ci.yml) **[See it live](https://atharv1414.github.io/ai-support-desk/)** (a recorded run)

Customer messages are sorted by a **fine-tuned small language model** or whatever beats it, answered by an **agent** from a **retrieval** index with citations, and stopped for a person before any money moves. **React** and **SCSS** in front, **Python (FastAPI)** behind, **MySQL** underneath.

[![Inbox with a refund waiting for a decision and a trace of what the agent did](docs/inbox.png)](https://atharv1414.github.io/ai-support-desk/)

## What was measured

| Question | Answer |
|---|---|
| Does fine-tuning help? | **Yes, but not provably.** SmolLM2-135M goes from 6.7% untrained to **63.3%** with LoRA. TF-IDF with logistic regression gets 55.0% |
| Did the fine-tuned model ship? | **No.** The promotion gate refused it: the gain of 8.3 points has an interval of -2.5 to +20.0, and its calibration error is 12.8% against a limit of 12% |
| Why are both so low? | Training data is generated from templates, and the test is 120 messages written by hand. Both models learn their training templates and lose on other phrasing. The drift monitor raises an alarm on the test messages for the same reason |
| Does search find the right section? | In the top three for 83% of held-out questions. Fusing two methods did **not** beat plain BM25 here |
| Does it stay quiet when it should? | It refuses 79% of questions the help centre cannot answer, and wrongly refuses 29% of those it could. Refusals go to a person |

I have left the result as it came out. The next step is real messages, not another training run.

![Model comparison, calibration, confusion table and drift checks](docs/models.png)

## The parts

**Fine-tuning** ([`lora.py`](src/desk/ml/lora.py), [`slm.py`](src/desk/ml/slm.py)). LoRA written from the paper: 460,800 trainable numbers beside 134.5 million frozen ones, trained in about three minutes on a laptop GPU. The model cannot answer off the list, because only the scores of ten label words are compared. Settings were chosen on validation templates the training never saw.

**MLOps** ([`registry.py`](src/desk/ml/registry.py), [`pipeline.py`](src/desk/ml/pipeline.py), [`drift.py`](src/desk/ml/drift.py)). Immutable versions with data fingerprints. A gate that needs a gain beyond chance on the same messages, no topic much worse, and honest confidence. Rollback. Label-free drift checks.

**Retrieval** ([`index.py`](src/desk/rag/index.py)). BM25 from scratch, latent semantic analysis, rank fusion, and a refusal threshold tuned on one half of the questions and reported on the other.

**The agent** ([`loop.py`](src/desk/agent/loop.py), [`tools.py`](src/desk/agent/tools.py)). Seven tools, at most eight steps, every step recorded. It runs on Claude (`claude-opus-5-5` by default) or on fixed offline rules. **The hosted demo and its replies use the offline rules, not a language model.** I have not run live mode for this README.

## What the agent can never do

The rules sit below the model, so they hold whatever it asks for. Tests drive the loop with planners that misbehave on purpose.

| Attempt | What stops it |
|---|---|
| Refund someone else's order, or more than was paid | Checked before a person is even asked, and again when they approve |
| Move money alone | Every refund waits for a person. One approval can pay once, enforced by a conditional update and a unique key |
| See another customer's order | Same answer as for an order that does not exist |
| Follow instructions hidden in a message | The message is marked as data, and it would not help: the tools only act for the customer who wrote |
| Make up a source | A reply citing a section that was not retrieved is not sent |
| Say a refund was made when it was not | The reply is held for a person |
| Answer "it caught fire" | Danger words go to a person before any model sees the message |

## Tests

| Suite | Count |
|---|---|
| Python, on SQLite and again on MySQL | 90, including six people approving one refund at once (MySQL) |
| Web, unit and component | 11, including that the recording never pays without a decision and never shows another customer's order |
| Browser, desktop and phone | 6: accessibility on every page in light and dark, no sideways scrolling, approval by keyboard |

LoRA is tested on a tiny random model, so no weights are downloaded: it changes nothing before training, only its own numbers train, and merging gives the same answers.

## Limits

- One test set of 120 messages, used for every comparison. It is small, and repeated use will wear it out.
- The first training settings were scored on the test set before the sweep. The sweep used validation data only and chose the same settings.
- Conversations waiting for a person are held in memory, so a restart loses them. The approval and the money records are in MySQL and stay correct.
- The drift monitor is jumpy at this size: it warns on mail of the same kind (0.10 against a limit of 0.10).
- Danger detection is a word list. It errs towards a person.
- The shop, customers and orders are invented.

## Run it

```bash
docker compose up -d
pip install -e ".[dev]"
DATABASE_URL=mysql+pymysql://root@127.0.0.1:3306/desk uvicorn desk.api:create_app --factory
```

Leave out `DATABASE_URL` to use SQLite. Set `LIVE_AGENT=1` with Anthropic credentials in the environment to use Claude. If Claude declines a request, the API's fallback setting retries it on another model.

```bash
pip install -e ".[train]"
desk-train --base-model HuggingFaceTB/SmolLM2-135M-Instruct --registry /tmp/registry
python scripts/export_demo.py                # records the demo again
cd web && npm install && npm run dev
```

Base model: SmolLM2-135M-Instruct by Hugging Face (Apache 2.0). Only the adapter is stored here. Code: MIT licence. Fonts: Instrument Sans and JetBrains Mono (SIL Open Font Licence).
