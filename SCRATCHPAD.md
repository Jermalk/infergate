## Carried over:

Session 14 addressed round 5 from ov_server: added Router.cache_stats() (hit/miss counters on _EmbedCache), RouteDecision.estimated_cost_usd (computed in select_model() from winner.cost_per_1k_tokens * estimated_tokens / 1000), and typed RouteTrace.scope_source as Literal with default "global". All changes non-breaking. select_model() return type widened to 4-tuple; all call sites updated. 128/128 tests pass. v0.1.9 published to PyPI. SIGNAL.md set to RELEASE READY v0.1.9.
