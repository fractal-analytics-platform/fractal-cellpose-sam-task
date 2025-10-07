from pathlib import Path

import pytest
from ngio import ChannelSelectionModel, create_synthetic_ome_zarr

from fractal_cellpose_sam_task.cellpose_sam_segmentation_task import (
    cellpose_sam_segmentation_task,
)
from fractal_cellpose_sam_task.utils import (
    IteratorConfiguration,
    MaskingConfiguration,
)


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 10, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((1, 1, 10, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task(
    tmp_path: Path, shape: tuple[int, ...], axes: str
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
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path), threshold=18252, channel=channel, overwrite=False
    )

    # Check that the label image was created
    assert "DAPI_0_thresholded" in ome_zarr.list_labels()

    label = ome_zarr.get_label("DAPI_0_thresholded")
    label_data = label.get_as_numpy()
    # Check that the label image is not empty
    assert label_data.max() > 0
    # DISCLAIMER: This is only a very basic test.
    # More comprehensive tests should be implemented based on the expected
    # results not only the presence of a label image.


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 10, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((1, 1, 10, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task_masked(
    tmp_path: Path, shape: tuple[int, ...], axes: str
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
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="nuclei_mask"),
        roi_table=None,
    )
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        threshold=18252,
        channel=channel,
        overwrite=False,
        iterator_configuration=iter_config,
    )

    # Check that the label image was created
    assert "DAPI_0_thresholded" in ome_zarr.list_labels()

    label = ome_zarr.get_label("DAPI_0_thresholded")
    label_data = label.get_as_numpy()
    # Check that the label image is not empty
    assert label_data.max() > 0
    # DISCLAIMER: This is only a very basic test.
    # More comprehensive tests should be implemented based on the expected
    # results not only the presence of a label image.
