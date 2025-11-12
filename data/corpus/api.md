# API

The `frobnicate(text, level)` function accepts:
- text: string
- level: integer 1–5; default 3

Returns a JSON object with:
- `result`: string
- `confidence`: float in [0, 1]

## Batch mode

Use `frobnicate_many(texts, level)` to process a list of texts.
Batching improves throughput and may reduce cost/latency.
