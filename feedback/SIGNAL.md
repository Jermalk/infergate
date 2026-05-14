# Feedback Loop Signal

**Direction:** RELEASE READY
**Last updated:** 2026-05-14
**infergate version on PyPI:** 0.1.3
**infergate version in ov_server venv:** 0.1.2 (wheel — upgrade needed)
**Last round file:** round_02_v0.1.2.md
**Last addressed file:** addressed_02_v0.1.3.md

---

## Valid states

| State | Who sets it | What the other session does |
|---|---|---|
| `INITIALIZED` | setup only | ov_server: begin integration round 1 |
| `FEEDBACK READY` | ov_server session | infergate session: read round file, implement, ship to PyPI |
| `RELEASE READY` | infergate session | ov_server session: upgrade venv, start next round |

---

## Protocol

### ov_server session → writing feedback

1. Complete an integration round (or hit a meaningful blocker worth reporting).
2. Copy `ROUND_TEMPLATE.md` → `round_NN_vX.Y.Z.md` (zero-padded round number, current infergate version).
3. Fill in all sections — especially "explicit non-requests" and "upgrade delta".
4. Update this file: set Direction, Last updated, Last round file.

```bash
# Check current installed version:
source /home/jerzy/ov_env/bin/activate && pip show infergate | grep Version
```

### infergate session → after shipping

1. Read the round file named in "Last round file".
2. Implement changes, run tests, publish to PyPI.
3. Copy `RESPONSE_TEMPLATE.md` → `addressed_NN_vX.Y.Z.md` (round number + NEW version).
4. Update this file: set Direction → RELEASE READY, new PyPI version, date.

### ov_server session → on RELEASE READY

```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version   # confirm new version
```

Then start the next integration round.
