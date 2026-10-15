# =============================================================================
# task_prompt.py -- ONE place that decides what language string the model sees.
#
# ---------------------------------------------------------------------------
# WHY THIS MODULE EXISTS
#
# The prompt has to be identical in three places or the policy silently runs
# out of distribution:
#
#   1. TRAINING   convert_v30_to_v21.py writes it into meta/tasks.jsonl, and
#                 the config sets prompt_from_task=True, so that file IS the
#                 training prompt.
#   2. SERVING    serve_behavior_policy.py passes it as default_prompt.
#   3. DIAGNOSING diagnose_prompt_sensitivity.py compares candidate strings.
#
# Every one of those used to carry its OWN hardcoded copy. That is how we lost
# a 30,000-step run: the converter wrote the task NAME, and nothing cross-
# checked it. A mismatch produces no error, no warning and no log line -- just
# a policy that scores badly for reasons that look like bad hyperparameters.
#
# ---------------------------------------------------------------------------
# THE RULE: READ IT FROM DISK, DO NOT TYPE IT IN
#
# The authoritative string is published by the challenge in
#   behavior-1k/2026-challenge-demos :: meta/tasks.jsonl
# and fetch_task_demos.py already copies it into each task's
# fetch_manifest.json as "task_description". Both files land on the pod during
# a normal rebuild, so at runtime there is always a real source to read.
#
# The constant below is a LAST RESORT for running off-pod, and it is
# deliberately marked unverified -- see the warning on it.
# =============================================================================

from __future__ import annotations

import json
import pathlib

# The literal snake_case string boxing_books_lora_v1 was trained on, because
# convert_v30_to_v21.py wrote task_name instead of task_description. Kept as a
# named constant because serving that checkpoint REQUIRES reproducing it.
V1_BUGGED_PROMPT = "boxing_books_up_for_storage"

# VERIFIED 2026-08-15 against the live upstream file:
#   huggingface.co/datasets/behavior-1k/2026-challenge-demos
#     -> meta/tasks.jsonl, task_index 23
#
# This settles a disagreement between two of our own notes. day10:89 transcribed
# it as "Put all six books from the bookshelves into the cardboard box"; that
# transcription is WRONG and has been corrected. The string below is what
# upstream actually says.
#
# It is still only a FALLBACK. Everything that matters -- training, serving,
# diagnosing -- resolves from the dataset on disk, because the dataset is what
# training actually reads and a constant here can always drift from it.
FALLBACK_PROMPT_TASK_23 = (
    "Put all six books from the bookcases in the living room "
    "into the box on the living room floor."
)

# Kept as an alias so nothing breaks if an older call site still refers to it.
UNVERIFIED_FALLBACK_PROMPT = FALLBACK_PROMPT_TASK_23

TASK_INDEX = 23
TASK_NAME = "boxing_books_up_for_storage"


def read_converted_dataset_tasks(dataset_root: str | pathlib.Path) -> list[dict] | None:
    """Every row of a CONVERTED v2.1 dataset's meta/tasks.jsonl, in file order.

    This file is the highest-authority source at training time, because it is
    literally what openpi reads when prompt_from_task=True.

    Since the multi-task converter (2026-09-23) each row carries four keys:

        task_index          our DENSE index, 0..N-1 -- what the frames join on
        task                the prompt, the string the model is handed
        source_task_index   the UPSTREAM challenge index (77, 89, 92)
        task_name           the upstream snake_case name

    Datasets converted before that date have only the first two. Everything
    here degrades to "this is a single-task dataset" in that case rather than
    guessing at the missing keys.

    Returns None if the dataset or the file is not present -- callers decide
    whether that is fatal.
    """
    tasks_file = pathlib.Path(dataset_root) / "meta" / "tasks.jsonl"
    if not tasks_file.is_file():
        return None
    rows = []
    for line in tasks_file.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows or None


def read_converted_dataset_prompt(
    dataset_root: str | pathlib.Path, task_index: int = 0
) -> str | None:
    """The prompt at one DENSE task_index of a converted dataset.

    task_index here is OUR index into the dataset's own task table (0, 1, 2),
    not the upstream challenge index. It defaults to 0, which is the whole
    table for a single-task dataset -- so every pre-cluster call site keeps its
    exact previous behaviour.

    To ask "what is upstream task 89's prompt", use prompt_for_source_task().
    """
    rows = read_converted_dataset_tasks(dataset_root)
    if rows is None:
        return None
    for row in rows:
        if row.get("task_index") == task_index:
            return row["task"]
    return None


def prompt_for_source_task(
    dataset_root: str | pathlib.Path, source_task_index: int
) -> str | None:
    """The prompt for an UPSTREAM task index (77, 89, 92) in a converted dataset.

    This is the lookup the cluster needs, and it is deliberately unwilling to
    guess. A single-task dataset converted from task 23 has exactly one prompt,
    and returning it for a query about task 89 would be the worst class of bug
    this project has: a real sentence, correctly formatted, describing the
    wrong task, handed to the model with nothing logged.

    So:
      - if the rows carry source_task_index, match on it and return None when
        nothing matches;
      - if they do not (a pre-cluster dataset), the question is unanswerable --
        return None rather than offering up the only prompt on file.
    """
    rows = read_converted_dataset_tasks(dataset_root)
    if rows is None:
        return None
    for row in rows:
        if row.get("source_task_index") == source_task_index:
            return row["task"]
    return None


def describe_converted_dataset(dataset_root: str | pathlib.Path) -> str:
    """A one-block summary of a dataset's task table, for logs and error text."""
    rows = read_converted_dataset_tasks(dataset_root)
    if rows is None:
        return f"{dataset_root}: no meta/tasks.jsonl"
    lines = [f"{dataset_root}/meta/tasks.jsonl -- {len(rows)} task(s):"]
    for row in rows:
        upstream = row.get("source_task_index")
        name = row.get("task_name", "?")
        tag = f"upstream task {upstream} ({name})" if upstream is not None else "upstream index not recorded"
        lines.append(f"    [{row.get('task_index')}] {tag}")
        lines.append(f'        "{row.get("task")}"')
    return "\n".join(lines)


def read_manifest_prompt(demos_dir: str | pathlib.Path) -> str | None:
    """The prompt as copied from upstream by fetch_task_demos.py.

    fetch_manifest.json carries "task_description", taken verbatim from the
    challenge's own meta/tasks.jsonl. This is the source convert_v30_to_v21.py
    uses, so reading it here lets us predict what a reconvert WILL produce
    before spending the ~48 minutes to produce it.
    """
    manifest = pathlib.Path(demos_dir) / "fetch_manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text()).get("task_description")


def read_upstream_prompt(
    tasks_jsonl: str | pathlib.Path, task_index: int = TASK_INDEX
) -> str | None:
    """The prompt straight from the challenge's own meta/tasks.jsonl.

    fetch_checkpoint.py downloads this file to resolve task -> checkpoint, so a
    copy is usually in the HuggingFace cache on the pod. Highest authority of
    all, and the only source that can settle the notes/day10 disagreement.
    """
    path = pathlib.Path(tasks_jsonl)
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_index") == task_index:
            return row["task"]
    return None


def resolve_prompt(
    *,
    override: str | None = None,
    dataset_root: str | pathlib.Path | None = None,
    demos_dir: str | pathlib.Path | None = None,
    task_index: int | None = None,
) -> tuple[str, str]:
    """Return (prompt, where_it_came_from), most authoritative source first.

    Order is deliberate: the converted dataset outranks the manifest, because
    the dataset is what training actually reads -- if the two disagree, the
    dataset is the truth about this checkpoint even when the manifest is more
    "correct" in the abstract.

    An explicit override always wins, so serving the old v1 checkpoint stays a
    one-flag operation.

    task_index is the UPSTREAM challenge index this run is for (23, 77, 89 ...),
    or None when the caller has not said. It controls two things:

      - whether a MULTI-TASK dataset can be resolved at all. It cannot be
        resolved without one: a cluster dataset holds three prompts and picking
        the first is how you evaluate task 89 while telling the model to install
        a modem. That failure looks completely plausible in a video.
      - whether the built-in task-23 fallback is legal (it is not, for any other
        task).

    ⚠️ A SINGLE-TASK DATASET STILL RESOLVES WITHOUT ONE, deliberately. There is
    no ambiguity to resolve, and requiring the argument would break every
    pre-cluster call site for no safety gain.
    """
    if override is not None:
        return override, "--prompt override"

    if dataset_root is not None:
        rows = read_converted_dataset_tasks(dataset_root)
        source = f"{pathlib.Path(dataset_root)}/meta/tasks.jsonl"

        if rows is not None and len(rows) > 1:
            # Multi-task: the caller MUST say which task, and it must be there.
            if task_index is None:
                raise ValueError(
                    f"{source} holds {len(rows)} tasks, so 'the prompt' is not a "
                    "single string. Pass task_index (the upstream challenge "
                    f"index) to choose one.\n{describe_converted_dataset(dataset_root)}"
                )
            found = prompt_for_source_task(dataset_root, task_index)
            if found is None:
                raise ValueError(
                    f"{source} has no row for upstream task {task_index}.\n"
                    f"{describe_converted_dataset(dataset_root)}\n"
                    "Refusing to fall back to another task's prompt -- that is a "
                    "real sentence describing the wrong job, and nothing "
                    "downstream would report it."
                )
            return found, f"{source} (upstream task {task_index})"

        if rows is not None and len(rows) == 1:
            row = rows[0]
            # If the row knows its upstream index and the caller named a
            # different one, this dataset is not the one they think it is.
            upstream = row.get("source_task_index")
            if task_index is not None and upstream is not None and upstream != task_index:
                raise ValueError(
                    f"{source} was converted from upstream task {upstream} "
                    f"({row.get('task_name', '?')}), but this run is for task "
                    f"{task_index}. Serving one task's prompt against another "
                    "task's dataset is the exact failure this module exists to "
                    "prevent."
                )
            return row["task"], source

    if demos_dir is not None:
        found = read_manifest_prompt(demos_dir)
        if found is not None:
            return found, f"{pathlib.Path(demos_dir)}/fetch_manifest.json"

    # The built-in fallback describes TASK 23 ONLY. Handing it to any other task
    # would silently prompt the model with a completely unrelated instruction --
    # far worse than the snake_case bug, because it reads as a valid sentence and
    # nothing about it looks wrong in a log. Now that we run more than one task
    # (p23 books, p77 modem), refuse rather than guess.
    #
    # task_index=None means "the caller did not say", which is how every
    # pre-cluster call site invokes this. That keeps the old behaviour -- fall
    # back, and let warn_if_unverified() shout about it -- rather than turning a
    # signature default into a silent behaviour change. Naming a task that is
    # not 23 is still refused outright.
    if task_index is not None and task_index != TASK_INDEX:
        raise ValueError(
            f"No prompt found on disk for task {task_index}, and the built-in "
            f"fallback describes task {TASK_INDEX} ({TASK_NAME}) only.\n"
            "Point this at the converted dataset or the demos dir, or pass an "
            "explicit override -- do NOT let it fall back."
        )

    return FALLBACK_PROMPT_TASK_23, "built-in fallback (task 23)"


def warn_if_unverified(source: str, log) -> None:
    """Shout if we ended up on the hardcoded string rather than a real file.

    The string itself is verified (see FALLBACK_PROMPT_TASK_23). What is worth
    warning about is that we are NOT reading the file training reads, so any
    drift between the two is invisible from here.
    """
    if source.startswith("built-in fallback"):
        log("=" * 72)
        log("USING THE BUILT-IN FALLBACK PROMPT -- no dataset or manifest found.")
        log("The string is verified against upstream, but it is NOT the file")
        log("training reads, so drift between the two cannot be detected here.")
        log("Point this at a converted dataset if the run matters.")
        log("=" * 72)
