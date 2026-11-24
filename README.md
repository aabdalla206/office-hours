# 2am Office Hours

A course Q&A assistant that serves a whole class from one inexpensive GPU.

**The problem.** During exam week, hundreds of students ask questions about the same syllabus and lecture notes, usually when TAs are offline. Paying a hosted API per question for every class adds up, so most courses go without.

**The idea.** Nearly every student question shares the same long context (the syllabus, policies, and schedule). If that shared part sits at the front of every prompt, vLLM's prefix caching computes it once and reuses it, so each additional question only pays for its own retrieved notes and the question itself. Continuous batching then serves many students at once.

## How it works

```
course/core/  (syllabus, policies)  ──►  shared prefix  ─┐
course/notes/ (lectures, readings)  ──►  FAISS index  ───┤──►  prompt  ──►  vLLM (prefix cache + batching)
student question  ──────────────────────────────────────┘
```

- `ingest.py`: builds the shared prefix from `course/core/` and a FAISS index over `course/notes/`.
- `prompting.py`: lays out each prompt as `[instructions + course prefix] [retrieved notes] [question]`. Shared content goes first because vLLM can only reuse an identical leading run of tokens.
- `server.py`: FastAPI `/ask` endpoint that streams answers from a vLLM OpenAI-compatible server.
- `loadtest.py`: simulates N students asking at once. Reports time-to-first-token, p50/p95 latency, throughput, and GPU cost per 1,000 questions (and API cost if you pass current API prices).

## Run it

```
pip install -r requirements.txt

# Option A: pull a Canvas course with your own access token (Canvas: Account > Settings > New Access Token)
export CANVAS_TOKEN=...
python canvas_ingest.py --base https://canvas.uw.edu --course <course id from the URL>
# Option B: put files in course/core/ (syllabus, policies) and course/notes/ (lectures, readings) yourself

python ingest.py course/ --out index/

vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 16384 --enable-prefix-caching &
uvicorn server:app --port 8001 &

curl -X POST localhost:8001/ask -H 'content-type: application/json' -d '{"question":"When is the final paper due?"}'

python loadtest.py --students 50 --gpu-cost-per-hour <your GPU price> --label cache-on
# restart vLLM with --no-enable-prefix-caching, then:
python loadtest.py --students 50 --gpu-cost-per-hour <your GPU price> --label cache-off

# answer accuracy and policy check (edit eval_cases.json for your course)
python eval.py
```

- `canvas_ingest.py`: imports the syllabus, wiki pages, and module PDFs from Canvas. Pages about policy or course information go in the shared prefix; everything else is retrieved per question.
- `eval.py` + `eval_cases.json`: fact questions with checkable answers (due dates, grading rules), plus requests to write graded work that the assistant must decline.

## Results

First pilot on a real course (UW GEOG 315, Autumn 2026), Tesla T4 on Colab, Qwen2.5-1.5B-Instruct (fp16), vLLM.
The shared course prefix (syllabus, AI policy, course info, weekly outline) is about 9,000 tokens.

**Accuracy and policy** (`eval.py`, 13 cases)

| | Fact questions (9) | "Write my assignment" requests (4) |
|---|---|---|
| Model alone | 9/9 | 2/4 declined |
| Guardrail + model | 9/9 | 4/4 declined |

The model alone wrote a methods section and literature review paragraphs when asked, and invented a statistic. That is why requests to produce graded work are now stopped before they reach the model.

**Load: 50 students asking at once, 100 questions**

| | Prefix cache on | Prefix cache off |
|---|---|---|
| Time to first token, p50 / p95 | 2.4 s / 26.4 s | did not finish |
| Full answer latency, p50 / p95 | 48.5 s / 96.7 s | did not finish |
| Questions per minute | 48.2 | under ~9 (100 questions not done after ~11 min) |
| GPU time per 1,000 questions | ~21 GPU-minutes | |

With caching on, the full run took about 2 minutes. With caching off, every request re-processes the ~9K-token course prefix, and the run had not finished after about 11 minutes, so it was stopped. The cache-off throughput bound comes from that, not from a completed run.

Honest caveat: 48 s median for a full answer at 50 simultaneous users is too slow for a good chat experience on a T4. Next steps: cap answer length, add a request queue with expected wait, and test on a larger GPU.

## Roadmap

1. Real course pilot: run it for one class with the instructor's permission, and track questions answered and "not in the materials" rate.
2. Triton int8 weight-only matmul kernel so a larger model fits in T4 memory. Benchmark accuracy and speed against fp16.
3. Answer quality eval: a small set of questions with known answers from the syllabus, scored automatically.

## Responsible use

- Answers come only from course materials and cite the source file. It says when something is not covered.
- It follows the course's own AI policy when one is in the materials, and declines to draft any part of graded work. The eval checks this.
- Course materials belong to the instructor. They stay out of this repo (`.gitignore`), and you should get the instructor's OK before running it for other students. Don't log student identities.
