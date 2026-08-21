# Hermes SEO Workflow
<!-- HERMES_SEO_AUTOPILOT -->

Work only in the open project root: the directory that contains this file and
`seo_workflow.py`.

## Rule 1 — review import has priority

If the user asks to import a reviewed workbook, apply `Correct Intent` values,
or save knowledge, run exactly this one command and report its JSON result:

```text
"./.venv/Scripts/python.exe" "./seo_workflow.py" apply-review "<reviewed-file.xlsx>" --quiet
```

This command reads the permanent ASCII `workflow_job_id` from the workbook and
finds the correct job itself. Never pass `--job`. Never run cleaning,
classification, clustering, `next`, `run`, `run-auto`, `feedback`, or `learn`
for a review-import request.

## Rule 2 — normal cleaning workflow

For a new cleaning request or after compaction, run:

```text
"./.venv/Scripts/python.exe" "./seo_workflow.py" next --input "<file>" --topic "<topic>" --quiet
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

Use one intent contract throughout the job. `commercial` means that the user
seeks a concrete offer/result or conversion: finding or choosing a product,
service, provider, vacancy, employer, rental, admission, download, booking, or
other actionable result. It does not require a payment verb and covers both
sides of a marketplace. A vacancy search/application and an employer hiring
query are both commercial. `informational` means explanation, duties,
instructions, diagnostics, reference facts, requirements discussed as
reference, reviews, comparisons, or general education without seeking a
concrete offer/result. Question words never decide intent. `garbage` is outside
the stated topic/business scope. Never redefine these classes midway through a
job.

If `next` requests `label_intent_family_batch`, run its `family-review`
command. Each row is a frequent lexical or two-token structural family plus real source examples.
Review all shown contexts; they are selected for diversity rather than file order.
Return every listed ID as `ID|commercial`, `ID|informational`, or `ID|neutral`
in `after_family_labels`. Use a decisive class only when the family itself
determines intent across the examples. Reviews, opinions, user experience,
pros/cons, instructions, diagnostics, reference facts, specifications,
diagrams, comparisons, and explanations are informational in this workflow's
binary output. Transactions and conversion actions are commercial. Brands,
products, topic nouns, weak question words, and mixed families are neutral.
Copy every ID, replace only the example decisions, execute immediately, and
call `next` again.

If `next` requests `label_unrepresented_family_examples`, execute its
`family-coverage-review` command and label every returned full phrase with the
same job-wide `commercial` / `informational` / `garbage` contract. These are
real source rows reserved to prevent a frequent or medium-frequency family
from reaching the final classifier without any phrase-level training example.
Execute `after_labels` immediately and call `next` again.

For `relevance-review` batches, decide topic membership before intent. A phrase
is `garbage` when it is outside the user's stated business scope even if it
looks transactional or contains a brand/topic word. The batch intentionally
mixes semantic outliers and representatives of different microclusters; judge
each row independently and never force a fixed class ratio.

If `next` requests `label_cluster_relevance_batch`, run its `cluster-review`
command. Judge each cluster against the user's exact business scope. For the
main `RC...` cluster ID use only `relevant`, `garbage`, or `mixed`. Brand-name
or transactional wording is not evidence of topic relevance. Use `mixed` only
when the central representatives genuinely span both scopes. Every
`relevance_risk` or `cluster_boundary` representative with its own `review_id`
must also receive an individual `relevant` or `garbage` decision, even when the
whole cluster is relevant. Copy every ID listed in `required_decisions` into
the compact `ID|label` list; do not omit or invent IDs. Replace only the
example decisions in `after_cluster_labels`, execute it immediately, and call
`next` again.

If `next` requests `define_intent_policy`, `refine_intent_signals`, or
`refine_final_intent_policy`, execute
its `policy-context` command. During refinement, use the returned coverage
metrics and replace the complete policy once; do not append duplicate signals.
The final refinement is regression-guarded: the workflow keeps the existing
policy automatically if the candidate worsens reviewed-example coverage or
false positives. Accept that compact result and continue with `next`.
From the labeled examples and the user's business scope, create 3–8 explicit
commercial prototypes, 5–12 implicit commercial structures without buy/price
words, 5–12 informational prototypes, 8–20 strong commercial signals, 12–30
strong informational signals, 3–10 weak question signals, 5–12 broad
relevant-topic prototypes, and 8–15 diverse hard-negative garbage prototypes.
Generate these signals for the current topic; never assume an automotive topic.
Strong commercial signals independently express concrete offer/result seeking,
a transaction, or a conversion. Commercial intent does not require a payment
verb; implicit marketplace demand must be represented too.
Strong informational signals independently express instructions, diagnostics,
reference data, specifications, diagrams, comparisons, or explanation. Weak
question words such as where/how/how much/which are never decisive alone:
phrases equivalent to "where to buy" and "how much does it cost" are commercial.
Use the returned weak-question examples to create separate strong context
signals for transaction/conversion and informational reference/location
meanings. Use a trailing `*` only for productive stems of at least four
characters and one concept per signal. Replace every signal reported as
rejected; never reuse a short or label-conflicting wildcard.
Implicit commercial structures
must represent bare transactional noun phrases typical for this topic. Hard negatives
must include plausible lexical collisions with the topic as well as unrelated
entities, media/stories, jobs/services, and other meanings of topic words.
They are synthetic boundary examples, never source rows. Replace only the eight
quoted prototype lists in `after_policy`, separate items with semicolons, then
execute it immediately and call `next` again.

The informational policy must also cover the current language's equivalents
of reviews, opinions, user experience, pros/cons, and product/service
overviews. Reuse decisive reviewed intent families as signals; never turn a
neutral family into a marker.

Do not ask the user to continue and do not send progress messages between steps.

## Long compute runs

`run`, `run-auto`, `run-large`, and `apply-cluster-decisions` may take longer than the terminal's
foreground timeout. This timeout does **not** mean that the workflow failed.
For any of these returned long commands, start it once in the terminal's background mode
with completion notification enabled. After `Background process started`, do
not call `wait`, `poll`, `status`, `next`, or another run command, and do not
send progress messages. End the turn immediately. When the terminal sends its
completion notification, call `next --job "<job>" --quiet` exactly once.

If `next` returns `{"status":"running", ...}`, a workflow process is
already active. Do not retry it, do not poll it, and do not wait in a loop.
End the turn immediately. Only a terminal completion notification may resume
the workflow. A non-run tool error may be handled by calling `next` once.

The workflow uses only the project-local `../models/multilingual-e5-base`
embedding model. Never download or replace it during a task. It selects CUDA
automatically when the installed PyTorch can use an NVIDIA GPU, otherwise CPU.

## Command safety

- Copy every workflow command literally. Do not construct a PowerShell command,
  add a PowerShell prefix, replace paths, or route Hermes through `seo.ps1`.
- The quoted project interpreter plus `seo_workflow.py` is the only approved
  Python entry point. Never call system Python, `python -c`, pip, pandas, a
  sandbox, or any other `.py` script directly.
- Never read XLSX, CSV, logs, or temporary files directly; use only the workflow
  commands and their compact JSON responses.
- Never convert, re-encode, copy, or create a replacement CSV. The maintained
  workflow detects UTF-8, CP1251, and CP866 itself, handles one-column query
  exports without treating commas inside a query as delimiters, and stages a
  large XLSX internally when necessary. Its temporary staging file is deleted
  by the workflow itself.
- Do not call `status` during a normal workflow.
- Inputs and reviewed workbooks are in `files\`; jobs are in `jobs\`; results
  are in `outputs\`. `README.md` is for humans only.
