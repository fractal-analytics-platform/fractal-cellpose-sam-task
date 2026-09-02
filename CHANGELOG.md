# Changelog

## v0.3.0

- Migrate to ngio 1.1 (`ngio>=1.1.0,<1.2`, up from `ngio>=0.5.8,<0.6`) and to
  fractal-tasks-utils 0.2.0, which is built on the same ngio line.
- Replace the private `ngio.images._image._parse_channel_selection` import
  with the public `Image.resolve_channel_selection`, added in ngio 1.1 for
  exactly this `skip_if_missing` check.

## v0.1.8

- Fix inconsistent relabelling caused by empty frames (#16). Empty frames
  (or ROIs) no longer reset the label counter, preventing duplicate labels
  across time points.
