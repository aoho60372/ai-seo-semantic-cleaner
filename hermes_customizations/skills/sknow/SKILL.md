---
name: sknow
description: Validate and import one reviewed SEO workbook without recalculating semantic results.
---

# Save SEO knowledge

Interpret the text after `/sknow` as exactly one reviewed Excel filename from
`E:\AI\seo\files`.

If the filename is absent, contains more than one filename, or is ambiguous,
do not run a command. Return only:

```text
Usage: /sknow <reviewed-file.xlsx>
Example: /sknow semantic_clustered_reviewed.xlsx
```

Otherwise run exactly:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" apply-review "<reviewed-file.xlsx>" --quiet
```

The workflow validates that the workbook exists and reads its permanent
`workflow_job_id` to find the correct job. Never pass `--job`. Never call
`next`, `run`, `run-auto`, `feedback`, or `learn`. Report only the resulting
JSON.
