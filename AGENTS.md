# AGENTS.md

## Project Instructions

- This repository is a Python speech emotion recognition project using PyTorch
  for model code and RAVDESS audio data. Treat data loading, audio transforms,
  tensor shapes, train/eval mode, randomness, metrics, model serialization, and
  filesystem paths as important boundaries.
- Planned tooling: Python with `uv` for environment/dependency workflow,
  `ruff` for lint/format, and `pytest` for tests. This migration is not
  configured yet. Do not assume these commands work, and do not present them as
  current project commands until the migration is done.
- This file is for agent working rules. Keep human usage and onboarding in
  `README.md`. After the tooling migration, add the small set of current agent
  commands here.
- Keep project-specific ML/audio guidance in this section so the general
  section can be copied to other projects.
- Prefer small real-data slices and simulations for development feedback before
  expensive full-pipeline runs. Use focused tests for deterministic data
  transformations or regressions that are easy to check.
- Separate loading from transformation, modeling, and metric logic so the parts
  that fiddle with data can be checked without mocks.
- Assert important ML boundaries: dataset metadata, file existence, tensor
  dimensions, tensor dtypes, class counts, train/eval state, model outputs, and
  saved/loaded model assumptions.

## General Coding Style

These are strong preferences, not blind absolutes. Bend them when the problem
demands it, but keep the exception small and intentional.

### Minimum Code Ladder

Before writing code, climb this ladder:

1. Understand the real problem and flow.
2. YAGNI: does this need to exist? Only build what is needed now.
3. Reuse existing code and patterns.
4. Use standard library, native platform features, or already-installed
   dependencies when they fit.
5. KISS: make the simplest working version.
6. Prefer plain functions and explicit data flow. Use classes only when shared
   state or a real domain concept earns them.
7. Prefer composition over inheritance.
8. Make the simple version readable.
9. Add assertions at meaningful boundaries.
10. Add abstraction only when the code has earned it.

### Abstraction

- Start simple and with almost no abstractions.
- Build what is needed now. Avoid speculative extension points.
- Do not abstract at the first or second repetition unless the duplication is
  already causing confusion or risk.
- After the third real repetition, consider extracting a function or concept.
- Prefer composition over inheritance. Keep inheritance rare because it couples
  behavior through extension and override.
- Use objects only when they represent a real concept with shared state and
  procedures that operate on that state.

### Intentional Programming

- Call functions only when their preconditions are already met.
- Do not call a broad function and make it sort out unrelated cases internally
  when the caller can choose the right path first.
- Add assertions to catch unintentional calls early.
- Prefer procedural pipelines: load data, transform data, model data, report
  results. Keep side effects visible and near the boundary.
- Minimize global state. Prefer simple functions that are easy to check in
  isolation, even when they are not tested yet.

### Naming

- Use meaningful, descriptive names.
- Avoid abbreviations unless they are standard in the domain.
- Do not encode types in names when real types can express them.
- Include units in names when values have units, such as `duration_sec`,
  `sample_rate_hz`, or `timeout_ms`.

### Nesting

- Avoid more than three levels of indentation.
- Use inversion: return early for invalid or finished cases.
- Use extraction: move a coherent block into a function when nesting hides the
  main flow.

### Comments And Docstrings

- Use docstrings for public functions, classes, CLI entry points, data
  transforms, and non-trivial helpers.
- Docstrings should describe contracts: purpose, inputs, outputs, units, shapes,
  invariants, side effects, and failure cases.
- Use inline comments for why, tradeoffs, and surprising domain decisions.
- Avoid comments that merely restate what the code does. If the code is unclear,
  improve the names or structure.

### Assertions

- Use assertions as executable assumptions, especially at boundaries.
- Prefer paired assertions: check incoming data near the start and outgoing data
  before returning or writing.
- Assert positive space and negative space: what must be true and what must not
  accidentally be true.
- When an assertion protects a distant assumption, consider adding an equivalent
  assertion at the distant caller or writer.
- Do not fiddle blindly with data. Make expected shape, range, keys, and units
  explicit where mistakes would be expensive or confusing.

### Runnable Checks

- Do not start by writing tests unless the problem is already well understood.
  First build, run, inspect, and learn.
- Use runnable checks for speed and confidence, not ritual coverage.
- Prefer small simulations and real-data slices over mocks for development
  feedback.
- Add focused tests when they replace slow manual checks, protect fiddly data
  transformations, capture regressions, or make iteration faster.
- Use fuzz checks or randomized simulations when behavior matters across ranges
  of valid and invalid input.
- Avoid mocks by separating data loading from data transformation. Check the
  transformation directly with plain in-memory data.
- Do not chase 100% coverage or arbitrary coverage percentages. Test what makes
  mistakes likely, expensive, or hard to see manually.
