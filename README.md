# fractal-cellpose-sam-task

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![CI (build and test)](https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task/actions/workflows/build_and_test.yml)

A [Fractal](https://fractal-analytics-platform.github.io/) task to segment OME-Zarr images with [Cellpose](https://github.com/MouseLand/cellpose) (Cellpose-SAM models).

## Features
- 2D and 3D images, with or without time.
- Segments using 1 to 3 image channels.
- 3D images are segmented either in 3D or by segmenting each Z-plane and stitching the masks.
- Choice of Cellpose normalization (default, none or custom) and optional pre-/post-processing.
- Optionally creates a masking ROI table from the new label.

## Models
| Model setting | Model |
|---|---|
| Pretrained, `cpsam` (default) | The original Cellpose-SAM model |
| Pretrained, `cpsam_v2` | Updated Cellpose-SAM, with fewer spurious masks in low-contrast regions. |
| Custom | Cellpose weights from a file path, e.g. a model fine-tuned from `cpsam` |

Cellpose downloads pretrained weights from [Hugging Face](https://huggingface.co/mouseland/cellpose-sam) the first time a model is used. They are stored in `~/.cellpose/models`, or in the folder set by the `CELLPOSE_LOCAL_MODELS_PATH` environment variable.

The DINO-based models of Cellpose ≥4.2 (`cpdino`, `cpdino-vitb`) are not yet supported. They need Meta's `dinov3` package, which is not on PyPI and is released under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).

## Licensing
This task is released under the BSD 3-Clause license. The models downloaded by Cellpose are subject to the terms listed below, check that they fit your use.

- **Code**: Cellpose is BSD 3-Clause; its dependency `segment-anything` is Apache 2.0.
- **Pretrained weights** (`cpsam`, `cpsam_v2`): BSD 3-Clause, as declared on [Hugging Face](https://huggingface.co/mouseland/cellpose-sam). They are derived from Meta's SAM ViT-L weights (Apache 2.0).
- **Training data**: all Cellpose models were trained on CC-BY-NC data.
