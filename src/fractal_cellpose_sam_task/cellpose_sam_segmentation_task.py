"""This is the Python module for my_task."""

import logging
from typing import Optional

import numpy as np
from ngio import ChannelSelectionModel, open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage
from pydantic import validate_call
from skimage.measure import label
from skimage.morphology import ball, dilation, disk, opening, remove_small_objects

from fractal_cellpose_sam_task.utils import IteratorConfiguration, MaskingConfiguration


def segmentation_function(
    int_img: np.ndarray, threshold: int, min_size: int = 50
) -> np.ndarray:
    """Example segmentation function.

    This function will need to be adapted to the specific segmentation method.

    Args:
        int_img (np.ndarray): Input image to be processed
        threshold (int): Threshold value for binarization
        min_size (int): Minimum size of objects to keep

    Returns:
        label_img (np.ndarray): Labeled image
    """
    # Thresholding the images
    binary_img = int_img >= threshold

    # Removing small objects
    cleaned_img = remove_small_objects(binary_img, min_size=min_size)
    # Opening to separate touching objects
    if cleaned_img.ndim == 2:
        selem = disk(1)
    else:
        selem = ball(1)
    opened_img = opening(cleaned_img, selem)

    # Optional: Dilation to restore object size
    dilated_img = dilation(opened_img, selem)

    # Labeling the processed image
    label_img = label(dilated_img, connectivity=1)
    assert isinstance(label_img, np.ndarray)
    return label_img.astype("uint32")


def load_masked_image(
    ome_zarr,
    masking_configuration: MaskingConfiguration,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logging.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name, masking_table_name=masking_table_name
    )
    return masked_image


@validate_call
def cellpose_sam_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channel: ChannelSelectionModel,
    label_name: Optional[str] = None,
    threshold: int,
    min_size: int = 50,
    # Iteration parameters
    iterator_configuration: Optional[IteratorConfiguration] = None,
    overwrite: bool = True,
) -> None:
    """Segment an image using a simple thresholding method.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channel (ChannelSelectionModel): Select the input channel to be used for
            segmentation.
        label_name (Optional[str]): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_thresholded".
        threshold (int): Threshold value to be applied.
        min_size (int): Minimum size of objects. Smaller objects are filtered out.
        iterator_configuration (Optional[IteratorConfiguration]): Advanced
            configuration to control masked and ROI-based iteration.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logging.info(f"{ome_zarr=}")

    if label_name is None:
        label_name = f"{channel.identifier}_thresholded"
    label = ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    logging.info(f"Output label image: {label=}")

    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image()
        logging.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=0,  # channel,
            axes_order="zyx",
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
        )
        logging.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channel,
            axes_order="zyx",
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logging.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logging.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logging.info(f"Iterator updated with ROI table: {iterator=}")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logging.info("Starting processing...")
    for image_data, writer in iterator.iter_as_numpy():
        label_img = segmentation_function(
            int_img=image_data, threshold=threshold, min_size=min_size
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = label_img.max()
        writer(label_img)
    # No need to call label.consolidate()
    # SegmentationIterator handles this
    logging.info(f"label {label_name} successfully created at {zarr_url}")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellpose_sam_segmentation_task)
