---
name: crisp-dm
description: Record progress in the CRISP-DM process. Use when starting or finishing a phase (business understanding, data understanding, preparation, modeling, evaluation, deployment) or when an evaluation result forces going back to an earlier phase.
---

# CRISP-DM logging

1. Identify the phase (1–6) the work belongs to. If unsure, re-read the phase
   table in `AGENTS.md`.
2. Make sure the phase artefact exists and is updated (see table in AGENTS.md).
3. Append an entry to `docs/crisp_dm_log.md` using exactly this template:

```markdown
### <YYYY-MM-DD> — Phase <N>: <phase name> — <short title>
- **What was done:** ...
- **Decision & why:** ...
- **Evidence:** <metric / file / dataset stat that supports the decision>
- **Loop-back:** <none | back to phase X because metric Y = Z>
- **Next step:** ...
```

4. Keep entries factual. A loop-back must cite the metric that triggered it and
   what will be changed (prompt version, weight, model, data cleaning rule).
5. Never rewrite old entries — the log is the history the report is built from.
