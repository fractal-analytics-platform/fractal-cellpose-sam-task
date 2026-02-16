# Changelog

## v0.1.8

- Fix inconsistent relabelling caused by empty frames (#16). Empty frames
  (or ROIs) no longer reset the label counter, preventing duplicate labels
  across time points.
