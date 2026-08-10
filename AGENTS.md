# Hermes SEO Workflow
<!-- HERMES_SEO_AUTOPILOT -->

Work only in the open project root: the directory that contains this file and
`seo_workflow.py`.

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

For a new cleaning request or after compaction, run:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" next --input "<file>" --topic "<topic>" --quiet
```

Then repeat until the returned JSON has `status` `stop`, `blocked`, or
`running`:

1. Execute the exact text in the returned `command` field.
2. If labels are requested, label only that batch with `commercial`,
   `informational`, or real `garbage`. Replace only the example labels in the
   returned `after_labels` command with the compact labels for that batch, then
   execute it immediately. Do not explain individual labels, narrate progress,
   inspect counts, or call any other tool between the batch and this command.
   Do not alter anything else in the command.
3. Call `next` again.

Do not ask the user to continue and do not send progress messages between steps.

## Long compute runs

`run`, `run-auto`, and `run-large` may take longer than the terminal's
foreground timeout. This timeout does **not** mean that the workflow failed.
For a returned run command, start it once in the terminal's background mode
with completion notification enabled. After `Background process started`, do
not call `wait`, `poll`, `status`, `next`, or another run command, and do not
send progress messages. End the turn immediately. When the terminal sends its
completion notification, call `next --job "<job>" --quiet` exactly once.

If `next` returns `{"status":"running", ...}`, a workflow process is
already active. Do not retry it, do not poll it, and do not wait in a loop.
End the turn immediately. Only a terminal completion notification may resume
the workflow. A non-run tool error may be handled by calling `next` once.

The workflow uses only the project-local `../models/multilingual-e5-small`
embedding model. Never download or replace it during a task. It selects CUDA
automatically when the installed PyTorch can use an NVIDIA GPU, otherwise CPU.

## Command safety

- Copy every workflow command literally. Do not construct a PowerShell command,
  add a PowerShell prefix, replace paths, or use `.seo.ps1`.
- Never call Python, pip, pandas, a sandbox, or a custom `.py` script directly.
- Never read XLSX, CSV, logs, or temporary files directly; use only the workflow
  commands and their compact JSON responses.
- Never convert, re-encode, copy, or create a replacement CSV. The maintained
  workflow detects UTF-8, CP1251, and CP866 itself, handles one-column query
  exports without treating commas inside a query as delimiters, and stages a
  large XLSX internally when necessary. Its temporary staging file is deleted
  by the workflow itself.
- Do not call `status` during a normal workflow.
- Inputs and reviewed workbooks are in `files\`; jobs are in `jobs\`; results
  are in `outputs\`. `README.txt` is for humans only.
