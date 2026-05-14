## Carried over:

Session 8 processed round 4, which is a confirmation round — no code changes. ov_server confirmed Router.reselect() and RouteDecision.task_directive working correctly. router.py in ov_server is now 23 lines (startup/health state only). Two latent friction items noted (reselect calling available_models() synchronously; profile-name/force_tier vocabulary coupling) — both safe for now, no action needed. addressed_04_v0.1.4.md written, SIGNAL.md flipped to RELEASE READY. Next session opens with feedback/SIGNAL.md check.
