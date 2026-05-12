## Carried over:

Phase 3 demo gateway built and live-tested. Three demo requests show signal (O1 hashtag directive),
embedding (cosine similarity classification), and keyword (string-match signal) routing. Both backends
verified live: Ollama 13 models, OVH 21 models. Key fix: config.yaml `model_preference: fast` must be
`fastest` to match selector.py `_pick()` — "fast" falls through to `_best` and always picks last pool item.
