import logging
from pathlib import Path

import numpy as np
import pytest
from ngio import OmeZarrContainer, Roi, create_synthetic_ome_zarr
from ngio.tables import RoiTable
from skimage.metrics import adapted_rand_error

from fractal_cellpose_sam_task.cellpose_sam_segmentation_task import (
    cellpose_sam_segmentation_task,
)
from fractal_cellpose_sam_task.utils import (
    AdvancedCellposeParameters,
    CellposeChannels,
    IteratorConfiguration,
    MaskingConfiguration,
)

# Set logging level to DEBUG for detailed output during tests
# logging.basicConfig(level=logging.INFO)


def test_cellpose_sam_segmentation_task(
    tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the cellpose segmentation task."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=True,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    it_config = IteratorConfiguration(roi_table="well_ROI_table")

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        level_path="0",
        channels=channel,
        overwrite=True,
        iterator_configuration=it_config,
    )

    # Check that the label image was created


def test_cellpose_sam_segmentation_task_2(
    tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the cellpose segmentation task."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=True,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    it_config = IteratorConfiguration(roi_table="well_ROI_table")
    advanced_config = AdvancedCellposeParameters(verbose=True)

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        level_path="0",
        channels=channel,
        overwrite=True,
        iterator_configuration=it_config,
        advanced_parameters=advanced_config,
    )

    # Check that the label image was created


print("--- Running non Verbose ---")
test_cellpose_sam_segmentation_task(
    tmp_path=Path("./tmp/test2"), shape=(2, 1, 64, 64), axes="czyx"
)


print("--- Running Verbose ---")
test_cellpose_sam_segmentation_task_2(
    tmp_path=Path("./tmp/test2"), shape=(2, 1, 64, 64), axes="czyx"
)

print("--- Running on 20 Planes ---")
test_cellpose_sam_segmentation_task(
    tmp_path=Path("./tmp/test2"), shape=(20, 1, 64, 64), axes="tcyx"
)
