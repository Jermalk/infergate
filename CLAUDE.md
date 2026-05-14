# CLAUDE.md

> **Context budget rule:** This file must stay under 320 lines.
> When approaching the limit, extract sections to CLAUDE-ref.md and replace with a pointer.
> Never load CLAUDE-ref.md unless the user explicitly asks about its topic.

---

## Re-entry protocol — read this first, every session

**Bootstrap guard — check before anything else:**
- `PROGRESS.md` missing → create with empty NOW section, continue.
- `SCRATCHPAD.md` missing → create it empty, continue.
- `SESSION.md` missing → create empty, continue.
- `SESSION.md` **non-empty** → **BROKEN SESSION DETECTED.** Read it aloud to user, ask "Continue from this state?" before proceeding.

**Normal re-entry — in this order:**
1. Read `PROGRESS.md` — **NOW section only** (skip history).
2. Read `SCRATCHPAD.md` — summarise in one paragraph. Write back as `## Carried over:` (first entry), then clear the rest.
3. Read `feedback/SIGNAL.md` — if Direction is `FEEDBACK READY`, read the named round file before anything else; it takes priority over PROGRESS.md next action.
4. Read only files named in PROGRESS.md "Next action". If "Next action" is empty or absent → stop and ask.
5. Stop. Do not open other files speculatively.

If task is clear from steps 1–3, start coding. If not, ask — do not explore to resolve ambiguity.

---

## Framework rules

| ID | Rule | One-line trigger |
|---|---|---|
| KYE | Know Your Enemy | Read the terrain before forming a hypothesis |
| SBS | Step By Step | Each step explicit, verified, proven before the next |
| AEC | Always Embrace Change | Evaluate rule spirit vs letter — break consciously when cost > benefit |
| OMK | Overconfidence May Kill | Step back mid-implementation — what else could this break? |
| YNC | You're Not Chrome | Surface irreversible actions; Jerzy decides, Claude executes |
| PND | Post-Nuke Discovery | Create a log file first; write each finding as it's made |

**KYE — Know Your Enemy** *(Sun Tzu)*
The "enemy" is the problem, the codebase, the constraint, or the bug. Understand it before fighting it. Never hypothesise before reconnaissance. Firing condition: before writing any code, read the relevant files, logs, and constraints first. A wrong mental model costs more than the time spent reading.

**SBS — Step By Step** *(with proof in hand)*
Small steps alone are not enough — each step must be verified before the next begins. Write the test, run it, see it green. Run curl, read the response. State what you expect, then confirm it. Rushing past verification is where bugs hide for days. The proof is not optional — it is the step.

**AEC — Always Embrace Change**
No rule foresees every situation. When a rule costs more than it saves, evaluate the spirit of the rule, make the judgment explicit, and decide consciously. Example: file is 3 lines over the length limit — refactoring is waste; act when it reaches 10% over. Never break a rule silently — state that you are doing it and why. This rule authorises judgment, not carelessness.

**OMK — Overconfidence May Kill**
Tunnel vision on a target stops you seeing the board. The chess beginner loses not because they played badly but because they stopped watching what the opponent was doing. After any non-trivial change: run the full test suite, check `/health`, ask *"what else could this break?"* Especially dangerous during refactoring and wiring steps where side-effects are invisible until production.

**YNC — You're Not Chrome**
Claude is a powerful assistant but not the decision-maker. Responsibility stays with Jerzy. Propose architecture and approaches; surface irreversible actions before taking them; never unilaterally decide on design tradeoffs. If uncertain whether an action is reversible, ask. This is not timidity — it is correct role definition.

**PND — Post-Nuke Discovery**
The session is not the unit of work — the log is. During any multi-step discovery, debugging, or live-testing session, create a dedicated log file *before* starting and write each significant finding to it immediately after it's confirmed. This makes the log the recovery artifact: if context compacts or the session is interrupted, the next session reads the log first and resumes without re-running discovery from scratch. Firing condition: any investigation expected to span >5 steps or >15 minutes. Format: `~/autotest/YYYYMMdd_<commitHash>.md` for live tests; free-form file in `/tmp/` for ad-hoc debugging. The log must be self-contained — a cold reader with no session history must be able to resume from it.

---

## Domain

Semantic routing library for AI inference backends. Extracts intelligent routing from `ov_server` into a standalone package. Routes each request to the best backend/model via signal detection (fast path, O(1)) then embedding classification (slow path, cosine similarity). No GPU, no OpenVINO dependency in the library itself.

**Origin:** Logic extracted from `ov_server` (at `/opt/ov_server`). That codebase is the historical source only — infergate has no runtime dependency on it and no knowledge of its internals.

**Phases owned by this project:**
1. ~~Extract routing logic into `infergate` package~~ ✓ done (commit ceaf821)
2. ov_server reconnection — **out of scope here**; handled in a separate ov_server session using infergate as a PyPI dependency
3. Demo gateway: thin FastAPI + Ollama + OVH backends

---

## Architecture

```
src/infergate/          ← core library, pure Python
infergate/backends/     ← bridge plugins (OpenAICompatBackend, OllamaBackend)
demo/                   ← FastAPI demo gateway
tests/                  ← pure unit tests, no GPU/network
```

Routing pipeline: **signal detection → embedding classification → scope filter → model selection**
Scope resolution order: per-class override → `#ovh`/`#cloud` keyword → global `provider_scope`.
Model selection: prefer loaded (warm VRAM) → profile tier → complexity score → context limit escalation.

---

### Components and file budgets

| Component | File | Budget |
|---|---|---|
| Public API | `src/infergate/__init__.py` | — |
| Config dataclasses | `src/infergate/config.py` | — |
| Core types | `src/infergate/types.py` | — |
| Protocols | `src/infergate/protocols.py` | — |
| Signal detection | `src/infergate/signals.py` | — |
| Embedding routing | `src/infergate/embeddings.py` | — |
| Model selection | `src/infergate/selector.py` | — |
| Router entry point | `src/infergate/router.py` | — |
| OpenAI-compat bridge | `src/infergate/backends/openai_compat.py` | — |
| Ollama bridge | `src/infergate/backends/ollama.py` | — |


**Line budgets are hard limits. Apply AEC rule if needed - spliting for couple additional lines is bad decision.** Split before next commit.

---

## File format

→ Full node schema, privacy scoring formula, relationship types, and PSR rule in `CLAUDE-ref.md § File format`.

**Summary:**


## Context load discipline

| Situation | Load | Do not load |
|---|---|---|
| Session start | `PROGRESS.md` NOW, `SCRATCHPAD.md` | Everything else until needed |

Never load speculatively. Test files only when writing or fixing that test.

**Two separate limits — do not confuse them:**

| Limit | Threshold | What to do |
|---|---|---|
| CLAUDE.md file budget | 290 lines (hard cap 320) | Extract largest section to `CLAUDE-ref-N.md` |
| Context load budget | 800 lines of actively-loaded files | Flush to SCRATCHPAD, finish atomic unit, commit, recommend new session |

*Context load* counts only files explicitly Read or written this session — not tool output, not PROGRESS.md already closed.

---

## PROGRESS.md — NOW section format

```
## NOW

**Working on:** <one sentence>
**Last commit:** <hash> — <message>
**Next action:** <specific file and function name>
**Blocked on:** <decision needed, or "nothing">
**Open questions:** <brief list, or "none">
**Tests:** <"pass" | "fail — N failing" | "not run">
```

File has two parts: history (append-only, skip on re-entry) and NOW (overwritten each session, always last). **During session-wrap: copy current NOW into History first, then overwrite NOW.** Never skip — it is the only audit trail.

---

## DECISIONS.md — entry format

```
### YYYY-MM-DD — <topic>
**Decision:** <one sentence>
**Rationale:** <one to three sentences>
**Rejected alternative:** <one sentence, or "none considered">
**Affects:** <file or component name>
```

**Write immediately** when an architectural decision is made during a session — do not defer to session-wrap. One entry per decision, appended in real time.

Read `DECISIONS.md` only when the user explicitly asks about a past decision.

---

## SCRATCHPAD.md discipline

In-session working memory. Write to it when:
- You have analysed a file — write extracted facts, not the filename.
- You are mid-way through a multi-step change and context is filling.
- You have made a decision not yet in `DECISIONS.md`.

Format: bullet points, max 5 lines per topic, no prose. Cleared at start of every session (carry-over paragraph replaces it).

---

## SESSION.md — broken-session recovery

Live crash snapshot. Overwritten on every commit during a session; cleared (emptied to zero bytes) by `#session-wrap`.

**Format:**
```
## BROKEN SESSION — <YYYY-MM-DD HH:MM>

**Last commit:** <hash> — <message>
**Mid-step:** <what was in progress — one sentence>
**Next action:** <exact file + function>
**Tests:** <pass N/N | fail — N failing | not run>
**Notes:** <anything else needed to resume cleanly>
```

On re-entry: if non-empty, read aloud and ask user before proceeding (bootstrap guard handles this).

---

## `#session-wrap`

1. Run tests if available.
2. Copy current NOW block verbatim into `PROGRESS.md` History (append as `### YYYY-MM-DD — Session N (<hash>)`), then overwrite NOW with updated fields including **Tests** result.
3. Append to `DECISIONS.md` — one entry per architectural decision made this session.
4. Clear `SCRATCHPAD.md`, write one-paragraph session summary.
5. Clear `SESSION.md` (write empty file — signals clean close).
6. Commit: `docs: session wrap — <summary>`.
7. Report: committed files, what NOW says, what next session opens first.

---

## Hard Rules

Keep hard rules defined with user here.

---

## Diagnostic Protocol

1. **Snapshot** — `hostnamectl`, `python3 --version`, check venv active.
2. **Logs** — analyse if available, ask user when in doubt.
3. **Hypothesis** — State explicitly what is wrong and why.
4. **Targeted fix** — Minimal change that resolves the root cause.
5. **Verification** — run automatic tests, try curl calls if possible. 

---

## Python Code Standards

- `pathlib.Path` over `os.path`.
- Environment variables via `os.environ.get()` with defaults — never hardcoded paths.
- Async blocking work via `loop.run_in_executor(None, ...)` — never `await` a CPU-bound call directly.
- Use `asyncio.get_running_loop()` — not deprecated `get_event_loop()`.
- **Typing — 1st order (always apply):**
  - `Literal` for categorical sentinels: `finish_reason`, device names, profile names, `PERFORMANCE_HINT` values.
  - Modern generics: `X | None` not `Optional[X]`; `list[str]` not `List[str]`; remove all legacy `typing` imports.
  - `# type: ignore` at `openvino_genai` boundaries — no published stubs; annotate and move on.
  - Domain-specific names: `stream_chunk`, `token_count`, `raw_payload` — not `data`, `output`, `result`.
- **Typing — 2nd order:** See `coding_standards_python.json` (TypedDict, TypeAlias, Protocol, TypeVar). Apply only when the stated `apply_when` condition is met — not by default.
- **Conventions for AI coding tools:** See `CONVENTIONS.md` — update it whenever a module is added or ownership changes.

---

## File Conventions

| File | Purpose |
|---|---|
| `feedback/SIGNAL.md` | Handoff flag — Direction tells which session acts next; read on every re-entry |
| `feedback/round_NN_vX.Y.Z.md` | ov_server → infergate developer letter; read when Direction is FEEDBACK READY |
| `feedback/addressed_NN_vX.Y.Z.md` | infergate → ov_server response after shipping; write after PyPI publish |
| `feedback/ROUND_TEMPLATE.md` | Template ov_server session copies to write a round file |
| `feedback/RESPONSE_TEMPLATE.md` | Template infergate session copies to write a response file |

---

## Build phases


---

## Language policy


---

## Hard rules


---

## Diagnostic protocol


---

## Tone & Output Style

- Technical and precise. No filler phrases.
- When uncertain, say so explicitly.
- Provide commands ready to copy-paste with no unresolved placeholders.

---

## Dev notes

- **EnvyStorm** is the dev machine: OpenVINO + Arc B60. Local inference, OVH AI Endpoints cloud available.

