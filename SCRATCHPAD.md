## Carried over:

Session 6 acted on round 2 ov_server feedback. Two changes shipped in v0.1.3: (1) `routing_only: bool` property added to `Backend` Protocol and both concrete backends — lets `OVServerBackend` declare `routing_only=True` to self-document routing-only deployments without raising `NotImplementedError`; (2) `force_tier` comment on `InferRequest` clarified to distinguish programmatic override from message-directive routing. 79/79 tests pass. `addressed_02_v0.1.3.md` written, `SIGNAL.md` flipped to RELEASE READY. Next session opens with `feedback/SIGNAL.md` check.
