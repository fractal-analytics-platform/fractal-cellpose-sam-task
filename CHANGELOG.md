# Changelog

## v0.3.0

- Require cellpose 4.2 (`cellpose>=4.2,<5.0`, up from `>=4.0.6`) (#38).

### Changes to the task parameters:
- Add a `model` selector that replaces `custom_model` (#40):
  - mode `cpsam pretrained` picks a pretrained model, `cpsam` or `cpsam_v2` (new default);
  - mode `custom` takes the path to custom weights.
- Replace `do_3D`, `stitch_threshold`, `anisotropy` and `flow3D_smooth` with the `AdvancedCellposeParameters.segmentation_3d` mode selector (`3D` or `Stitch 2D planes`) (#40).
- Remove `bsize`, which had no effect with SAM models (#40).
- Remove top-level `invert`, which was ignored with `CustomNorm`; use `CustomNorm.invert` instead (#39, #40).

## v0.2.3

- Migrate to ngio 1.1 (`ngio>=1.1.0,<1.2`, up from `ngio>=0.5.8,<0.6`) and to
  fractal-tasks-utils 0.2.0, which is built on the same ngio line.
- Replace the private `ngio.images._image._parse_channel_selection` import
  with the public `Image.resolve_channel_selection`, added in ngio 1.1 for
  exactly this `skip_if_missing` check.

## v0.1.8

- Fix inconsistent relabelling caused by empty frames (#16). Empty frames
  (or ROIs) no longer reset the label counter, preventing duplicate labels
  across time points.
