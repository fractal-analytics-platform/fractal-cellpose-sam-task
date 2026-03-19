"""This is the Python module for my_task."""

import logging

import numpy as np
from cellpose import core, models
from fractal_tasks_utils.segmentation import (
    IteratorConfig,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from ngio import OmeZarrContainer, open_ome_zarr_container
from ngio.images._image import _parse_channel_selection
from ngio.utils import NgioValueError
from pydantic import Field, validate_call

from fractal_cellpose_sam_task.utils import (
    AdvancedCellposeParameters,
    AnyCreateRoiTableModel,
    CellposeChannels,
    CreateMaskingRoiTable,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger("cellpose_sam_task")


def _setup_cellpose_kwargs(
    is_3d: bool,
    calculated_anisotropy: float | None,
    cellpose_parameters: AdvancedCellposeParameters,
) -> dict:
    """Set up the keyword arguments for the Cellpose model evaluation.

    This function determines the appropriate parameters to pass to the
    Cellpose model based on whether 3D segmentation is being performed and
    whether an anisotropy factor has been calculated.
    """
    kwargs = cellpose_parameters.to_eval_kwargs()
    kwargs["z_axis"] = 1 if is_3d else None
    if not is_3d:
        # For 2D segmentation we need to set do_3D=False
        # to avoid having to add a single Z plane dimension to the
        # input and output
        kwargs["do_3D"] = False
    if (
        is_3d
        and not cellpose_parameters.do_3D
        and cellpose_parameters.stitch_threshold == 0.0
    ):
        raise ValueError(
            "For 3D images either do_3D must be set to True or "
            "if do_3D is False, stitch_threshold must be greater than 0.0."
        )
    kwargs["channel_axis"] = 0
    if cellpose_parameters.anisotropy is None:
        kwargs["anisotropy"] = calculated_anisotropy

    if cellpose_parameters.verbose:
        logger.info("Cellpose evaluation parameters:")
        for key, value in kwargs.items():
            logger.info(f" {key}: {value}")
    return kwargs


def segmentation_function(
    *,
    image_data: np.ndarray,
    model: models.CellposeModel,
    **kwargs,
) -> np.ndarray:
    """Wrap Cellpose segmentation call.

    Args:
        image_data (np.ndarray): Input image data
        model (models.CellposeModel): Preloaded Cellpose model.
        **kwargs: Additional keyword arguments to pass to the Cellpose model
            evaluation function

    Returns:
        np.ndarray: Segmented image
    """
    masks, _, _ = model.eval(
        image_data,
        **kwargs,
    )
    masks = np.expand_dims(masks, axis=0).astype(np.uint32)
    return masks


def _format_label_name(label_name_template: str, channel_identifier: str) -> str:
    """Format the label name based on the provided template and channel identifier.

    Args:
        label_name_template (str): The template for the label name. This
        might contain a placeholder "{channel_identifier}" which will be replaced
        by the channel identifier or no placeholder at all,
        in which case the channel identifier will be ignored.
        channel_identifier (str): The channel identifier to insert into the
            label name template.

    Returns:
        str: The formatted label name.
    """
    try:
        label_name = label_name_template.format(channel_identifier=channel_identifier)
    except KeyError as e:
        raise ValueError(
            "Label Name format error only allowed placeholder is "
            f"'channel_identifier'. {{{e}}} was provided."
        ) from e
    return label_name


def _skip_segmentation(channels: CellposeChannels, ome_zarr: OmeZarrContainer) -> bool:
    """Check wheter to skip the current task based on the channel configuration.

    If the channel selection specified in the channels parameter is not
    valid for the provided OME-Zarr image, this function checks the
    skip_if_missing attribute of the channels configuration.
    If skip_if_missing is True, the function returns True, indicating that the task
    should be skipped. If skip_if_missing is False, a ValueError is raised.

    Args:
        channels (CellposeChannels): The channel selection configuration.
        ome_zarr (OmeZarrContainer): The OME-Zarr container to check against.

    Returns:
        bool: True if the task should be skipped due to missing channels,
        False otherwise.

    """
    channels_list = channels.to_list()
    image = ome_zarr.get_image()
    try:
        _parse_channel_selection(image=image, channel_selection=channels.to_list())
    except NgioValueError as e:
        if channels.skip_if_missing:
            logger.warning(
                f"Channel selection {channels_list} is not valid for the provided "
                "image, but skip_if_missing is set to True. Skipping segmentation."
            )
            logger.debug(f"Original error message: {e}")
            return True
        else:
            raise ValueError(
                f"Channel selection {channels_list} is not valid for the provided "
                "image. If you want to skip processing when channels are missing, "
                "set skip_if_missing to True."
            ) from e
    return False


@validate_call
def cellpose_sam_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channels: CellposeChannels,
    label_name: str = "{channel_identifier}_segmented",
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfig | None = None,
    custom_model: str | None = None,
    # Cellpose parameters
    advanced_parameters: AdvancedCellposeParameters = Field(  # noqa: B008
        default_factory=AdvancedCellposeParameters
    ),
    pre_post_process: SegmentationTransformConfig = Field(  # noqa: B008
        default_factory=SegmentationTransformConfig
    ),
    create_masking_roi_table: AnyCreateRoiTableModel = Field(  # noqa: B008
        default_factory=SkipCreateMaskingRoiTable
    ),
    overwrite: bool = True,
) -> None:
    """Segment an image using Cellpose with SAM model.

    For more information, see:
        https://github.com/MouseLand/cellpose/tree/main/cellpose

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channels (CellposeChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        label_name (str): Name of the resulting label image. Optionally, it can contain
            a placeholder "{channel_identifier}" which will be replaced by the
            first channel identifier specified in the channels parameter.
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfig | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        custom_model (str | None): Path to a custom Cellpose model. If not
            set, the default "cpsam" model will be used.
        advanced_parameters (AdvancedCellposeParameters): Advanced parameters
            for Cellpose segmentation.
        pre_post_process (SegmentationTransformConfig): Configuration for pre- and
            post-processing transforms applied by the iterator.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    # Validate that the specified channels are present in the image
    if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
        return None

    # Format the label name based on the provided template and channel identifier
    label_name = _format_label_name(
        label_name_template=label_name, channel_identifier=channels.identifiers[0]
    )
    logger.info(f"Formatted label name: {label_name=}")

    # Determine if we are doing 3D segmentation
    # If so we need to set the anisotropy factor
    if ome_zarr.is_3d:
        image = ome_zarr.get_image(path=level_path)
        px_z, (px_y, px_x) = image.pixel_size.z, image.pixel_size.yx
        # Pixelsize must be isotropic in XY (to some extent)
        perc_diff_xy = abs(px_x - px_y) / max(px_x, px_y)
        if perc_diff_xy >= 0.01:
            logger.warning(
                f"Non-isotropic pixel size in XY detected: px_x={px_x}, px_y={px_y}"
            )
        px_xy = (px_x + px_y) / 2.0
        anisotropy = px_z / px_xy
        logger.info(
            "Anisotropy factor calculated: "
            f"(px_z={px_z} / px_xy={px_xy}) = {anisotropy}"
        )
    else:
        anisotropy = None

    # Initialize Cellpose model
    if custom_model is None:
        custom_model = "cpsam"

    model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=custom_model)

    if advanced_parameters.verbose:
        logging.getLogger("cellpose").setLevel(logging.INFO)
    else:
        logging.getLogger("cellpose").setLevel(logging.WARNING)
    cellpose_kwargs = _setup_cellpose_kwargs(
        is_3d=ome_zarr.is_3d,
        calculated_anisotropy=anisotropy,
        cellpose_parameters=advanced_parameters,
    )

    # Set up the segmentation iterator
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=channels.to_list(),
        output_label_name=label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        segmentation_func=lambda x: segmentation_function(
            image_data=x, model=model, **cellpose_kwargs
        ),
        iterator=iterator,
    )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Building a masking roi table
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        label = ome_zarr.get_label(name=label_name, path=level_path)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellpose_sam_segmentation_task)
