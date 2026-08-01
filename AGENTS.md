# Hermes SEO Workflow
<!-- HERMES_SEO_AUTOPILOT -->

Work only in `E:\AI\seo`.

## Rule 1 — review import has priority

If the user asks to import a reviewed workbook, apply `Correct Intent` values,
or save knowledge, run exactly this one command and report its JSON result:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" apply-review "<reviewed-file.xlsx>" --quiet
```

This command reads the permanent ASCII `workflow_job_id` from the workbook and
finds the correct job itself. Never pass `--job`. Never run cleaning,
classification, clustering, `next`, `run`, `run-auto`, `feedback`, or `learn`
for a review-import request.

## Rule 2 — normal cleaning workflow

For a new cleaning request, after compaction, or after a tool error, run:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" next --input "<file>" --topic "<topic>" --quiet
```

Then repeat until the returned JSON has `status` `stop` or `blocked`:

1. Execute the exact text in the returned `command` field.
2. If labels are requested, label only that batch with `commercial`,
   `informational`, or real `garbage`. Replace only the example labels in the
   returned `after_labels` command with the compact labels for that batch, then
   execute it. Do not alter anything else in the command.
3. Call `next` again.

Do not ask the user to continue and do not send progress messages between steps.

## Command safety

- Copy every workflow command literally. Do not construct a PowerShell command,
  add a PowerShell prefix, replace paths, or use `.seo.ps1`.
- Never call Python, pip, pandas, a sandbox, or a custom `.py` script directly.
- Never read XLSX, CSV, logs, or temporary files directly; use only the workflow
  commands and their compact JSON responses.
- Do not call `status` during a normal workflow.
- Inputs and reviewed workbooks are in `files\`; jobs are in `jobs\`; results
  are in `outputs\`. `README.txt` is for humans only.
