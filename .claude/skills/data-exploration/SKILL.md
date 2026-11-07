---
name: data-exploration
description: Profile a dataset and write CRISP-DM Data Understanding notes. Use when a new dataset is downloaded or a data quality question comes up.
---

# Data exploration

Goal: understand the data without dumping it into the agent context.

1. Load data only inside `scripts/explore_data.py` (add a function per dataset).
   Print aggregates, never full records. Samples: max 3 rows, text cut to 300 chars.
2. Always report:
   - size (rows, columns), dtypes, missing values per column
   - duplicates (exact and near-duplicate text)
   - text length distribution (chars / tokens: min, p50, p95, max)
   - label / category distribution
   - language, encoding artefacts, HTML leftovers, PII present
3. Write findings to `docs/02_data_understanding.md`:
   source + licence, schema, stats table, quality issues, and **implications
   for preparation** (each issue → a concrete cleaning rule).
4. Log the phase with the `crisp-dm` skill.
