---
name: seo-automation
description: Use for autonomous semantic-core cleaning and review imports in the maintained SEO project.
---

# SEO automation

Work only in the open project root containing `AGENTS.md` and
`seo_workflow.py`. The project's `AGENTS.md` is the source of truth for the
state machine, labeling rules, long background runs, and terminal states.

For a new cleaning request, start exactly with:

```text
"./.venv/Scripts/python.exe" "./seo_workflow.py" next --input "<file>" --topic "<topic>" --quiet
```

For a reviewed-workbook import, follow Rule 1 in `AGENTS.md`; never restart
cleaning. During an active job, copy the workflow's returned `command` fields
literally. The quoted project interpreter plus `seo_workflow.py` is the only
approved Python entry point. Never route Hermes through `seo.ps1`, system
Python, `python -c`, pip, pandas, or a custom script.
