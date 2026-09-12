# Texmin retrieval evaluations

Create a private JSONL file containing representative English, Hindi, and Hinglish questions.
Each line should identify one or more source filenames that must be retrieved:

```json
{"question":"Which regulation specifies the ventilation requirement?","expected_sources":["MineRegulations1961"]}
```

Run the evaluation after every embedding, chunking, OCR, or retrieval change:

```bash
python evaluate_retrieval.py evals/golden.jsonl --top-k 5
```

Do not select a model from public benchmark scores alone. Promote a new index only when it
improves this corpus-specific recall without violating the latency budget.
