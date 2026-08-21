# Changelog

## Unreleased

- Migrate to ngio 1.1 (`ngio==1.1.0a2`, temporary exact pin until the final
  1.1.0 release) and to the ngio-1.1 branch of fractal-tasks-utils
  (temporary git-ref dependency until its next release). No code changes were
  required: the task's own ngio surface is stable 1.1 API, and all iterator
  usage goes through fractal-tasks-utils.

## v0.1.8

- Fix inconsistent relabelling caused by empty frames (#16). Empty frames
  (or ROIs) no longer reset the label counter, preventing duplicate labels
  across time points.
