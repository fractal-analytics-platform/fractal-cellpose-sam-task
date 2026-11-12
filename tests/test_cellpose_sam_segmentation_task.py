import inspect
from pathlib import Path

import cellpose.models
import numpy as np
import pytest
from fractal_tasks_utils.segmentation import IteratorConfig, MaskingConfig
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from fractal_tasks_utils.transforms import (
    GaussianBlurConfig,
    SizeFilterConfig,
)
from ngio import OmeZarrContainer, Roi, create_synthetic_ome_zarr
from ngio.tables import MaskingRoiTable, RoiTable
from skimage.metrics import adapted_rand_error

from fractal_cellpose_sam_task.cellpose_sam_segmentation_task import (
    cellpose_sam_segmentation_task,
)
from fractal_cellpose_sam_task.utils import (
    AdvancedCellposeParameters,
    CellposeChannels,
    CreateMaskingRoiTable,
    CustomNorm,
    SkipCreateMaskingRoiTable,
)

# Keep a reference to the real class before any test monkeypatches it, so that
# the mocks can validate their call arguments against the installed Cellpose API.
_RealCellposeModel = cellpose.models.CellposeModel
_REAL_INIT_SIGNATURE = inspect.signature(_RealCellposeModel.__init__)
_REAL_EVAL_SIGNATURE = inspect.signature(_RealCellposeModel.eval)


class _SignatureCheckedCellposeModel:
    """Base mock that rejects arguments the real CellposeModel would not accept.

    Calls are bound against the signatures of the installed
    `cellpose.models.CellposeModel`, so renamed or removed arguments raise
    `TypeError` exactly as they would with the real model.
    """

    def __init__(self, *args, **kwargs):
        _REAL_INIT_SIGNATURE.bind(self, *args, **kwargs)

    def eval(self, image, **kwargs):
        _REAL_EVAL_SIGNATURE.bind(self, image, **kwargs)
        return self._predict(image)

    def _predict(self, image):
        raise NotImplementedError


class MockCellposeModel(_SignatureCheckedCellposeModel):
    def _predict(self, image):
        masks = np.ones(image.shape[1:], dtype=np.uint32)
        return masks, None, None


class MockCellposeModelWithEmptyFrames(_SignatureCheckedCellposeModel):
    """Mock that returns empty masks on every other call to simulate empty frames."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0

    def _predict(self, image):
        self._call_count += 1
        if self._call_count % 2 == 0:
            # Return empty mask for even calls
            masks = np.zeros(image.shape[1:], dtype=np.uint32)
        else:
            # Return non-empty mask for odd calls
            masks = np.ones(image.shape[1:], dtype=np.uint32)
        return masks, None, None


def check_label_quality(
    ome_zarr: OmeZarrContainer, label_name: str, gt_name: str = "nuclei"
):
    if ome_zarr.is_3d:
        # Synthetic data is 2D only
        # we run 3D tests to check the API but cannot check label quality
        return
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.1, f"Adapted Rand Error too high: {are}>0.1. Labels might be wrong."


@pytest.mark.parametrize(
    "mock_cls", [MockCellposeModel, MockCellposeModelWithEmptyFrames]
)
def test_mock_rejects_arguments_unknown_to_cellpose(mock_cls):
    """The mocks must fail like the real model on arguments Cellpose doesn't accept."""
    with pytest.raises(TypeError):
        mock_cls(not_a_cellpose_argument=True)

    model = mock_cls(gpu=False, pretrained_model="cpsam")
    image = np.zeros((1, 8, 8), dtype=np.float32)
    with pytest.raises(TypeError):
        model.eval(image, not_a_cellpose_argument=True)

    masks, _, _ = model.eval(image, channel_axis=0)
    assert masks.shape == (8, 8)


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
        ((2, 1, 1, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task(
    monkeypatch, is_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
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
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_fast:
        # Mock Cellpose model in fast mode to avoid downloading the model
        import cellpose.models

        monkeypatch.setattr(
            cellpose.models,
            "CellposeModel",
            MockCellposeModel,
        )

    pre_post = SegmentationTransformConfig(
        pre_process=[
            GaussianBlurConfig(sigma_xy=1.0),
        ],
        post_process=[SizeFilterConfig(min_size=10)],
    )
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
        pre_post_process=pre_post,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()


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
        ((2, 1, 1, 64, 64), "tczyx"),
    ],
)
def test_cellpose_sam_segmentation_task_masked(
    monkeypatch, is_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the cellpose segmentation task with a masking configuration."""
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
        overwrite=False,
        axes_names=axes,
    )
    channel = CellposeChannels(
        mode="label",
        identifiers=["DAPI_0"],
    )

    iter_config = IteratorConfig(
        masking=MaskingConfig(
            mode="Masking", masking_source="Label Name", identifier="nuclei_mask"
        ),
        roi_table=None,
    )

    if is_fast:
        # Mock Cellpose model in fast mode to avoid downloading the model
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
    assert "DAPI_0_segmented" in ome_zarr.list_labels()
    if is_fast:
        return


@pytest.mark.real_model
def test_cellpose_sam_segmentation_task_no_mock(monkeypatch, is_fast, tmp_path: Path):
    """Base test for the cellpose segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"
    channel_labels = ["DAPI_0"]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_fast:
        import cellpose.models

        monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path), channels=channel, overwrite=False
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()
    if not is_fast:
        check_label_quality(ome_zarr, "DAPI_0_segmented")


def test_roi_table_cropping_cellpose_sam_segmentation_task_no_mock(
    monkeypatch, is_fast, tmp_path: Path
):
    """Base test for the cellpose segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (2, 2, 64, 64)
    axes = "tcyx"
    channel_labels = ["DAPI_0", "DAPI_1"]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_fast:
        import cellpose.models

        monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path), channels=channel, overwrite=False
    )
    roi = Roi.from_values(
        slices={"t": (0, 1), "y": (0, 10), "x": (0, 10)}, name="crop_roi"
    )
    roi_table = RoiTable(rois=[roi])
    ome_zarr.add_table("well_ROI_table", roi_table, overwrite=True)

    it_config = IteratorConfig(roi_table="well_ROI_table")

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        level_path="0",
        channels=channel,
        overwrite=True,
        iterator_configuration=it_config,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()

    label_data = ome_zarr.get_label("DAPI_0_segmented").get_as_numpy(axes_order="tyx")
    assert np.all(label_data[1, :, :] == 0), "Non-cropped region should be empty"
    assert np.any(label_data[0, :, :] > 0), "Cropped region should have some labels"


@pytest.mark.real_model
def test_custom_norm_no_mock(monkeypatch, is_fast, tmp_path: Path):
    """Base test for the cellpose segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"
    channel_labels = ["DAPI_0"]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_fast:
        import cellpose.models

        monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    advanced_params = AdvancedCellposeParameters(normalization=CustomNorm())
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
        advanced_parameters=advanced_params,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()
    if not is_fast:
        check_label_quality(ome_zarr, "DAPI_0_segmented")


def test_cellpose_sam_segmentation_with_masking_roi_table(
    monkeypatch, is_fast, tmp_path: Path
):
    """Base test for the cellpose segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"
    channel_labels = ["DAPI_0"]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])

    if is_fast:
        import cellpose.models

        monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    label_name = "custom_label_name"
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
        label_name=label_name,
        create_masking_roi_table=SkipCreateMaskingRoiTable(),
    )
    assert f"{label_name}" in ome_zarr.list_labels()
    assert f"{label_name}_masking_ROI_table" not in ome_zarr.list_tables()

    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=True,
        label_name=label_name,
        create_masking_roi_table=CreateMaskingRoiTable(),
    )
    assert f"{label_name}" in ome_zarr.list_labels(), ome_zarr.list_labels()
    assert f"{label_name}_masking_ROI_table" in ome_zarr.list_tables()
    table = ome_zarr.get_table(f"{label_name}_masking_ROI_table")
    assert isinstance(table, MaskingRoiTable)
    if not is_fast:
        assert len(table.rois()) == 5
        roi = table.rois()[0]
        assert roi.name == "1"
        assert roi.label == 1
        assert roi.space == "world"
        assert [(s.axis_name, s.start, s.end) for s in roi.slices] == [
            ("x", 0.0, 3.9000000000000004),
            ("y", 0.0, 8.125),
            ("z", 0.0, 1.0),
        ]
    else:
        assert len(table.rois()) >= 1


def test_empty_frames_do_not_reset_labels(monkeypatch, tmp_path: Path):
    """Test that empty frames don't cause label counter to reset (issue #16)."""
    test_data_path = tmp_path / "data.zarr"
    # 3 time points, no channel dimension
    shape = (3, 64, 64)
    axes = "tyx"

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names=axes,
    )

    import cellpose.models

    monkeypatch.setattr(
        cellpose.models,
        "CellposeModel",
        MockCellposeModelWithEmptyFrames,
    )

    channel = CellposeChannels(mode="label", identifiers=["DAPI_0"])
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
    )

    label_data = ome_zarr.get_label("DAPI_0_segmented").get_as_numpy(axes_order="tyx")

    # Frame 0 (call 1, odd): non-empty, labels should be 1
    # Frame 1 (call 2, even): empty, all zeros
    # Frame 2 (call 3, odd): non-empty, labels should be 2 (not reset to 1)
    assert label_data[1].max() == 0, "Frame 1 should be empty"
    assert label_data[0].max() > 0, "Frame 0 should have labels"
    assert label_data[2].max() > 0, "Frame 2 should have labels"

    # The key assertion: labels in frame 2 must not duplicate labels in frame 0
    labels_t0 = set(np.unique(label_data[0])) - {0}
    labels_t2 = set(np.unique(label_data[2])) - {0}
    assert labels_t0.isdisjoint(labels_t2), (
        f"Duplicate labels across frames! T=0: {labels_t0}, T=2: {labels_t2}"
    )


def test_skip_if_missing_with_valid_channel(monkeypatch, tmp_path: Path):
    """When skip_if_missing=True and the channel exists, the task runs normally."""
    test_data_path = tmp_path / "data.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    channel = CellposeChannels(
        mode="label", identifiers=["DAPI_0"], skip_if_missing=True
    )
    cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
    )

    # Channel exists → task must have run and created the label
    assert "DAPI_0_segmented" in ome_zarr.list_labels()


def test_skip_if_missing_with_invalid_channel(monkeypatch, tmp_path: Path):
    """When skip_if_missing=True and the channel is absent, the task skips silently."""
    test_data_path = tmp_path / "data.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    # Request a channel that does not exist in the image
    channel = CellposeChannels(
        mode="label", identifiers=["GFP_0"], skip_if_missing=True
    )
    result = cellpose_sam_segmentation_task(
        zarr_url=str(test_data_path),
        channels=channel,
        overwrite=False,
    )

    # Task should return None without raising and without creating any label
    assert result is None
    assert "GFP_0_segmented" not in ome_zarr.list_labels()


def test_raise_if_missing_with_invalid_channel(monkeypatch, tmp_path: Path):
    """When skip_if_missing=False and channel is absent, a ValueError is raised."""
    test_data_path = tmp_path / "data.zarr"
    create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", MockCellposeModel)

    # skip_if_missing defaults to False
    channel = CellposeChannels(mode="label", identifiers=["GFP_0"])
    with pytest.raises(ValueError, match="GFP_0"):
        cellpose_sam_segmentation_task(
            zarr_url=str(test_data_path),
            channels=channel,
            overwrite=False,
        )
