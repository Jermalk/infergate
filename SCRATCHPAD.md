## Carried over:

Session 14 addressed round 5 from ov_server: added Router.cache_stats() (hit/miss counters on _EmbedCache), RouteDecision.estimated_cost_usd (winner.cost_per_1k_tokens * estimated_tokens / 1000), and RouteTrace.scope_source typed as Literal with default "global". 128/128 tests. v0.1.9 published to PyPI and pushed to GitHub. Adopted strict semver from v0.2.0: MINOR for new API, PATCH for fixes only — recorded in CLAUDE.md, DECISIONS.md, and memory. CI/CD tag-triggered publish discussed but deferred.
