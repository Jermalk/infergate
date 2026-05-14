## Carried over:

Session 9 implemented roadmap step 1+5 (v0.1.5). Two additive fields: (1) RouteDecision.estimated_tokens — already computed inside decide(), now surfaced; (2) ModelDescriptor.cost_per_1k_tokens: float | None — config-only metadata, parsed through from_dict(), no routing logic change. 6 new tests, 93/93 pass. Plan stored at plans/20260514_feature_roadmap.md. Next: step 4 (RouteTrace) — instrument decide() to capture eliminated candidates and scope source; opt-in via trace=False parameter.
