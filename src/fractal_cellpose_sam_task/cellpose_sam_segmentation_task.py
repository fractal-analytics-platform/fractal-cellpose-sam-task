"""This is the Python module for my_task."""

import logging
import time
from typing import Optional

import numpy as np
from cellpose import core, models
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage
from pydantic import validate_call

from fractal_cellpose_sam_task.utils import (
    AdvancedCellposeParameters,
    CellposeChannels,
    IteratorConfiguration,
    MaskingConfiguration,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def segmentation_function(
    *,
    image_data: np.ndarray,
    model: models.CellposeModel,
    parameters: AdvancedCellposeParameters,
    do_3D: bool,
    anisotropy: Optional[float] = None,
) -> np.ndarray:
    """Wrap Cellpose segmentation call.

    Args:
        image_data (np.ndarray): Input image data
        model (models.CellposeModel): Preloaded Cellpose model.
        parameters (AdvancedCellposeParameters): Advanced parameters for
            Cellpose segmentation.
        do_3D (bool): Whether to perform 3D segmentation.
        anisotropy (Optional[float]): Anisotropy factor for z-axis scaling.

    Returns:
        np.ndarray: Segmented image
    """
    z_axis = 1 if do_3D else None

    kwargs = parameters.to_eval_kwargs()
    if parameters.anisotropy is None:
        # Replace anisotropy only if not set in parameters
        kwargs["anisotropy"] = anisotropy
    masks, _, _ = model.eval(
        image_data,
        do_3D=do_3D,
        z_axis=z_axis,
        channel_axis=0,
        **kwargs,
    )
    masks = np.expand_dims(masks, axis=0).astype(np.uint32)
    return masks


def load_masked_image(
    ome_zarr,
    masking_configuration: MaskingConfiguration,
    level_path: Optional[str] = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (Optional[str]): Optional path to a specific resolution level.

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
    label_name: Optional[str] = None,
    level_path: Optional[str] = None,
    # Iteration parameters
    iterator_configuration: Optional[IteratorConfiguration] = None,
    custom_model: Optional[str] = None,
    # Cellpose parameters
    advanced_parameters: AdvancedCellposeParameters = AdvancedCellposeParameters(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment an image using Cellpose with SAM model.

    For more information, see:
        https://github.com/MouseLand/cellpose/tree/main/cellpose

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channels (CellposeChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        label_name (Optional[str]): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_segmented".
        level_path (Optional[str]): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (Optional[IteratorConfiguration]): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        custom_model (Optional[str]): Path to a custom Cellpose model. If not
            set, the default "cpsam" model will be used.
        advanced_parameters (AdvancedCellposeParameters): Advanced parameters
            for Cellpose segmentation.
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
        pix_size_z, pix_size_xy = label.pixel_size.z, label.pixel_size.yx
        assert pix_size_xy[0] == pix_size_xy[1], "Non-isotropic pixel size in XY"
        anisotropy = pix_size_z / pix_size_xy[0]
    else:
        axes_order = "cyx"
        anisotropy = None

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
            parameters=advanced_parameters,
            do_3D=ome_zarr.is_3d,
            anisotropy=anisotropy,
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
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellpose_sam_segmentation_task)
