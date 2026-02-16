"""Pydantic models for advanced iterator configuration."""

import logging
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian, median
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


class GaussianFilter(BaseModel):
    """Gaussian pre-processing configuration.

    Attributes:
        type (Literal["gaussian"]): Type of pre-processing.
        sigma_xy (float): Standard deviation for Gaussian kernel in XY plane.
        sigma_z (float | None): Standard deviation for Gaussian kernel in Z axis.
            If not specified, no smoothing is applied in Z axis.
    """

    type: Literal["gaussian"] = "gaussian"
    sigma_xy: float = Field(default=2.0, gt=0)
    sigma_z: float | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.sigma_z is not None:
                logger.warning(
                    "sigma_z is specified but the input image is 2D. Ignoring sigma_z."
                )
            return gaussian(image, sigma=self.sigma_xy)
        elif image.ndim == 3:
            sigma = (
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)  # type: ignore[call-arg] is correct
        elif image.ndim == 4:
            sigma = (
                0,
                self.sigma_z if self.sigma_z is not None else 0,
                self.sigma_xy,
                self.sigma_xy,
            )
            return gaussian(image, sigma=sigma)  # type: ignore[call-arg] is correct
        else:
            raise ValueError("Input to Gaussian filter image must be 2D, 3D, or 4D.")


class MedianFilter(BaseModel):
    """Median filter pre-processing configuration.

    Attributes:
        type (Literal["median"]): Type of pre-processing.
        size_xy (int): Size in pixels of the median filter in XY plane.
        size_z (int | None): Size in pixels of the median filter in Z axis.
            If not specified, no filtering is applied in Z axis.
    """

    type: Literal["median"] = "median"
    size_xy: int = Field(default=2, gt=0)
    size_z: int | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Median filter to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Filtered image.
        """
        if image.ndim == 2:
            if self.size_z is not None:
                logger.warning(
                    "size_z is specified but the input image is 2D. Ignoring size_z."
                )
            return median(image, footprint=np.ones((self.size_xy, self.size_xy)))
        elif image.ndim == 3:
            size = (
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        elif image.ndim == 4:
            size = (
                1,
                self.size_z if self.size_z is not None else 1,
                self.size_xy,
                self.size_xy,
            )
            return median(image, footprint=np.ones(size))
        else:
            raise ValueError("Input to median filter image must be 2D, 3D, or 4D.")


class HistogramEqualization(BaseModel):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) pre-processing

    configuration.

    Attributes:
        type (Literal["histogram"]): Type of pre-processing.
        kernel_size_xy (int | None): Shape of kernel in XY plane.
            By default, kernel_size is 1/8 of image height by 1/8 of its width.
        kernel_size_z (int | None): Shape of kernel in Z axis.
            By default, kernel_size is 1/8 of image height by 1/8 of its width.
        clip_limit (float | None): Clipping limit, normalized between 0 and 1
            (higher values give more contrast).
        nbins (int | None): Number of gray bins for histogram (“data range”).
    """

    type: Literal["histogram"] = "histogram"
    kernel_size_xy: int | None = None
    kernel_size_z: int | None = None
    clip_limit: float | None = Field(default=0.01, ge=0, le=1)
    nbins: int | None = 256

    def _build_kernel_size(self, image: np.ndarray) -> np.ndarray | None:
        """Build kernel size tuple based on image dimensions.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray | None: Kernel size tuple or None for default behavior.
        """
        if image.ndim == 2:
            if self.kernel_size_z is not None:
                logger.warning(
                    "kernel_size_z is specified but the input image is 2D. "
                    "Ignoring kernel_size_z."
                )
            kernel_size = (self.kernel_size_xy, self.kernel_size_xy)

        elif image.ndim == 3:
            kernel_size = (
                self.kernel_size_z,
                self.kernel_size_xy,
                self.kernel_size_xy,
            )

        elif image.ndim == 4:
            kernel_size = (
                1,
                self.kernel_size_z,
                self.kernel_size_xy,
                self.kernel_size_xy,
            )
        else:
            raise ValueError("Input to median filter image must be 2D, 3D, or 4D.")

        # Return None if any kernel size component is None (use scikit-image defaults)
        # Return np.array only if all values are specified
        return (
            np.array(kernel_size) if all(k is not None for k in kernel_size) else None
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Histogram equalization to the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Histogram equalized image.
        """
        kernel_size = self._build_kernel_size(image)

        return equalize_adapthist(
            image,
            kernel_size=kernel_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )


PreProcess = Annotated[
    GaussianFilter | MedianFilter | HistogramEqualization,
    Field(discriminator="type"),
]


class SizeFilter(BaseModel):
    """Size filter post-processing configuration.

    Attributes:
        type (Literal["size_filter"]): Type of post-processing.
        min_size (int): Minimum size in pixels for objects to keep.
    """

    type: Literal["size_filter"] = "size_filter"
    min_size: int = Field(ge=0)

    def apply(self, labels: np.ndarray) -> np.ndarray:
        """Apply size filtering to the labeled image.

        Args:
            labels (np.ndarray): Labeled image.

        Returns:
            np.ndarray: Size-filtered labeled image.
        """
        return remove_small_objects(labels, max_size=self.min_size)


PostProcess = Annotated[
    SizeFilter,
    Field(discriminator="type"),
]


class PrePostProcessConfiguration(BaseModel):
    """Configuration for pre- and post-processing steps.

    Attributes:
        pre_process (list[PreProcess]): List of pre-processing steps.
        post_process (list[PostProcess]): List of post-processing steps.
    """

    pre_process: list[PreProcess] = Field(default_factory=list)
    post_process: list[PostProcess] = Field(default_factory=list)


def apply_pre_process(
    image: np.ndarray,
    pre_process_steps: list[PreProcess],
) -> np.ndarray:
    """Apply pre-processing steps to the image.

    Args:
        image (np.ndarray): Input image.
        pre_process_steps (list[PreProcess]): List of pre-processing steps.

    Returns:
        np.ndarray: Pre-processed image.
    """
    for step in pre_process_steps:
        image = step.apply(image)
    return image


def apply_post_process(
    labels: np.ndarray,
    post_process_steps: list[PostProcess],
) -> np.ndarray:
    """Apply post-processing steps to the labeled image.

    Args:
        labels (np.ndarray): Labeled image.
        post_process_steps (list[PostProcess]): List of post-processing steps.

    Returns:
        np.ndarray: Post-processed labeled image.
    """
    for step in post_process_steps:
        labels = step.apply(labels)
    return labels
