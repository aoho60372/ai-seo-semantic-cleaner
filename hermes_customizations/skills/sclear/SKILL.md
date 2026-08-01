---
name: sclear
description: Start a validated autonomous SEO semantic-core cleaning workflow from a file and topic.
---

# SEO semantic cleaning

Interpret the text after `/sclear` as:

```text
<file> <topic>
```

The file is a filename from the open project's `files\` directory; the topic
is all remaining text. A filename containing spaces may be quoted.

If either value is absent or ambiguous, do not run a command. Return only:

```text
Usage: /sclear <file> <topic>
Example: /sclear semantic.xlsx Продажа автозапчастей
```

Otherwise run exactly:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ".\seo.ps1" next --input "<file>" --topic "<topic>" --quiet
```

The workflow validates that the input file exists. If it returns an error,
report that error and do not guess a filename or topic. If it returns JSON with
a `command` field, follow the normal autonomous workflow in `AGENTS.md`.
