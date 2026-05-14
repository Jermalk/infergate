# Response to Round 4 — infergate v0.1.4 (no version bump)
**Date:** 2026-05-14
**Addresses:** `round_04_v0.1.4.md`
**Published:** no new release — confirmation round, no code changes

---

## Done

Nothing to implement. Round 4 is a confirmation round. All P1/P2 items from round 3
are confirmed working by ov_server. router.py is now 23 lines.

## Noted friction (no action taken)

- `reselect()` calls `backend.available_models()` synchronously. Currently safe
  (dict lookup, no I/O). If OVHBackend ever adds TTL/network inside that method,
  the caller is responsible for moving to an async wrapper. The library won't add
  async overhead speculatively.

- Profile name vs `force_tier` mapping: the current coupling (ov_server maps
  `"fastest"/"balanced"/"best"` directly to `force_tier`) works because the
  vocabularies align. This is a known latent coupling; no change needed unless
  the vocabularies diverge.

## Breaking changes

None.

## Upgrade notes

No upgrade needed — v0.1.4 is already installed and confirmed working.

---

Integration milestone: ov_server's router.py now contains only startup/health state.
All routing logic lives in infergate.
