# LMM implementation — phase block diagrams (living log)

**Requirement (see [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md)):** after **each** implementation phase (**L-A … L-K** for LMM build, and **R-1 … R-5** when retiring pipeline B), append a **new dated section** below. Each section **must** include:

1. **Phase id** (e.g. `L-D`, `R-3`) and **date** / PR link.
2. **ASCII block graph** of the **currently implemented** pipeline (or the **delta** this phase added), in the same style as Figure A/B/C in the LMM plan (boxes, arrows). Mark **NEW** / **CHANGED** blocks explicitly.
3. **Placement check:** one short paragraph — does this block sit in the right place vs Figure C?
4. **Gap check:** bullet list — anything **missing** vs that phase’s **Acceptance checks** in the plan? Anything **extra** or wrong?

Older phases stay in this file for audit trail; do not delete past sections.

---

## Template (copy for each new phase)

```markdown
## Phase <L-X or R-X> — <title> — YYYY-MM-DD (<PR or branch>)

### Block graph (as implemented after this phase)

\`\`\`
(paste ASCII diagram here)
\`\`\`

### Placement vs [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md) Figure C

(text)

### Gap check vs plan acceptance

- [ ] …
```

---

## Log entries

_(No phases logged yet.)_
