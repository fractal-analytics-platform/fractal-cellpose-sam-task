"""This is the Python module for my_task."""

import logging
import time

import numpy as np
from cellpose import core, models
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage
from pydantic import validate_call

from fractal_cellpose_sam_task.pre_post_process import (
    PrePostProcessConfiguration,
    apply_post_process,
    apply_pre_process,
)
from fractal_cellpose_sam_task.utils import (
    AdvancedCellposeParameters,
    AnyCreateRoiTableModel,
    CellposeChannels,
    CreateMaskingRoiTable,
    IteratorConfiguration,
    MaskingConfiguration,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger(__name__)


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
    pre_post_process: PrePostProcessConfiguration,
    **kwargs,
) -> np.ndarray:
    """Wrap Cellpose segmentation call.

    Args:
        image_data (np.ndarray): Input image data
        model (models.CellposeModel): Preloaded Cellpose model.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.
        **kwargs: Additional keyword arguments to pass to the Cellpose model
            evaluation function

    Returns:
        np.ndarray: Segmented image
    """
    # Pre-processing
    image_data = apply_pre_process(
        image=image_data,
        pre_process_steps=pre_post_process.pre_process,
    )
    masks, _, _ = model.eval(
        image_data,
        **kwargs,
    )
    # Post-processing
    masks = apply_post_process(
        labels=masks,
        post_process_steps=pre_post_process.post_process,
    )
    masks = np.expand_dims(masks, axis=0).astype(np.uint32)
    return masks


def load_masked_image(
    ome_zarr,
    masking_configuration: MaskingConfiguration,
    level_path: str | None = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (str | None): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


@validate_call
def cellpose_sam_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channels: CellposeChannels,
    label_name: str | None = None,
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfiguration | None = None,
    custom_model: str | None = None,
    # Cellpose parameters
    advanced_parameters: AdvancedCellposeParameters = AdvancedCellposeParameters(),  # noqa: B008
    pre_post_process: PrePostProcessConfiguration = PrePostProcessConfiguration(),  # noqa: B008
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment an image using Cellpose with SAM model.

    For more information, see:
        https://github.com/MouseLand/cellpose/tree/main/cellpose

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channels (CellposeChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        label_name (str | None): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_segmented".
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        custom_model (str | None): Path to a custom Cellpose model. If not
            set, the default "cpsam" model will be used.
        advanced_parameters (AdvancedCellposeParameters): Advanced parameters
            for Cellpose segmentation.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")
    if label_name is None:
        label_name = f"{channels.identifiers[0]}_segmented"

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    label = ome_zarr.get_label(name=label_name, path=level_path)
    logger.info(f"Derived label image: {label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Determine if we are doing 3D segmentation
    # If so we need to set the anisotropy factor
    if ome_zarr.is_3d:
        axes_order = "czyx"
        px_z, (px_y, px_x) = label.pixel_size.z, label.pixel_size.yx
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
        axes_order = "cyx"
        anisotropy = None
    logger.info(f"Segmenting using {axes_order=}")

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image(path=level_path)
        logger.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=channels.to_list(),
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logger.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channels.to_list(),
            axes_order=axes_order,
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logger.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logger.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logger.info(f"Iterator updated with ROI table: {iterator=}")

    # Initialize Cellpose model
    # Check if colab notebook instance has GPU access
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
    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logger.info("Starting processing...")
    run_times = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)
    for it, (image_data, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        label_img = segmentation_function(
            image_data=image_data,
            model=model,
            pre_post_process=pre_post_process,
            **cellpose_kwargs,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = label_img.max()
        writer(label_img)
        iteration_time = time.time() - start_time
        run_times.append(iteration_time)

        # Only log the progress every logging_step iterations
        if it % logging_step == 0 or it == num_rois - 1:
            avg_time = sum(run_times) / len(run_times)
            logger.info(
                f"Processed ROI {it + 1}/{num_rois} "
                f"(avg time per ROI: {avg_time:.2f} s)"
            )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Building a masking roi table
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(name=table_name, table=masking_roi_table)
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellpose_sam_segmentation_task)
