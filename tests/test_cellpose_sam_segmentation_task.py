from pathlib import Path

import numpy as np
import pytest
from ngio import OmeZarrContainer, create_synthetic_ome_zarr
from skimage.metrics import adapted_rand_error

from fractal_cellpose_sam_task.cellpose_sam_segmentation_task import (
    cellpose_sam_segmentation_task,
)
from fractal_cellpose_sam_task.utils import (
    CellposeChannels,
    IteratorConfiguration,
    MaskingConfiguration,
)


class MockCellposeModel:
    def __init__(self, *args, **kwargs):
        pass

    def eval(self, image, **kwargs):
        masks = np.ones(image.shape[1:], dtype=np.uint32)
        return masks, None, None


def check_label_quality(
    ome_zarr: OmeZarrContainer, label_name: str, gt_name: str = "nuclei"
):
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.01, (
        f"Adapted Rand Error too high: {are}<0.01. Labels might be wrong."
    )


def check_masked_label_quality(
    ome_zarr: OmeZarrContainer,
    label_name: str,
    gt_name: str = "nuclei",
    masking_label_name: str = "nuclei_mask",
):
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    mask = ome_zarr.get_label(masking_label_name).get_as_numpy(axes_order="tzyx", t=0)

    ground_truth = ground_truth * (mask > 0)  # Apply mask to ground truth
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.01, (
        f"Adapted Rand Error too high: {are}<0.01. Labels might be wrong."
    )


@pytest.mark.skip()
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 2, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((2, 1, 2, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task(
    monkeypatch, is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the threshold segmentation task."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channel_labels=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_github_or_fast:
        # Mock Cellpose model in GitHub Actions to avoid downloading the model
        import cellpose.models

        monkeypatch.setattr(
            cellpose.models,
            "CellposeModel",
            MockCellposeModel,
        )

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path), channels=channel, overwrite=False
    )

    # Check that the label image was created
    assert "DAPI_0_thresholded" in ome_zarr.list_labels()


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 2, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((2, 1, 2, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task_masked(
    monkeypatch, is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the threshold segmentation task with a masking configuration."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channel_labels=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel = CellposeChannels(
        mode="label",
        identifiers=["DAPI_0"],
    )

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="nuclei_mask"),
        roi_table=None,
    )

    if is_github_or_fast:
        # Mock Cellpose model in GitHub Actions to avoid downloading the model
        import cellpose.models

        monkeypatch.setattr(
            cellpose.models,
            "CellposeModel",
            MockCellposeModel,
        )

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
        iterator_configuration=iter_config,
    )

    # Check that the label image was created
    assert "DAPI_0_thresholded" in ome_zarr.list_labels()
    if is_github_or_fast:
        return
    check_masked_label_quality(
        ome_zarr,
        "DAPI_0_thresholded",
    )


def test_cellpose_sam_segmentation_task_no_mock(monkeypatch, tmp_path: Path):
    """Base test for the threshold segmentation task."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channel_labels=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path), channels=channel, overwrite=False
    )

    # Check that the label image was created
    assert "DAPI_0_thresholded" in ome_zarr.list_labels()

    check_label_quality(ome_zarr, "DAPI_0_thresholded")
