"""Pydantic models for advanced iterator configuration."""

from typing import Literal, Optional

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field


class MaskingConfiguration(BaseModel):
    """Masking configuration.

    Args:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: Optional[str] = None


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration.

    Args:
        masking (Optional[MaskingIterator]): If set, the segmentation will be
            performed only within the confines of the specified mask. A mask can be
            specified either by a label image or a Masking ROI table.
        roi_table (Optional[str]): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    masking: Optional[MaskingConfiguration] = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: Optional[str] = Field(default=None, title="Iterate Over ROIs")


class NormalizationParameters(BaseModel):
    """Normalization parameters for Cellpose.

    The normalization is applied before running the Cellpose model for
    each channel independently.

    Args:
        mode: Literal["default", "custom", "none"]:
            - "default": use Cellpose default normalization (0-1 rescaling and 1%-99%
                percentile clipping). all other parameters are ignored.
            - "none": do not apply any normalization, and ignore all other parameters.
            - "custom": use custom normalization parameters as specified below.
        lowhigh (Optional[tuple[float, float]]): pass in normalization values for
            0.0 and 1.0 as list [low, high]
        sharpen (float): sharpen image with high pass filter, recommended to be
            1/4-1/8 diameter of cells in pixels
        normalize (bool): run normalization (if False, all following parameters ignored)
        percentile (tuple[float, float]): pass in percentiles to use as list
            [perc_low, perc_high]
        tile_norm_blocksize (int): compute normalization in tiles across image to
            brighten dark areas, to turn on set to window size in pixels (e.g. 100)
        norm3D (bool): compute normalization across entire z-stack rather than
            plane-by-plane in stitching mode.
    """

    mode: Literal["default", "custom", "none"] = "default"
    lowhigh: tuple[float, float] = (0.0, 1.0)
    sharpen: float = 0.0
    normalize: bool = True
    percentile: tuple[float, float] = (1.0, 99.0)
    tile_norm_blocksize: int = 0
    norm3D: bool = False

    def to_eval_kwargs(self) -> bool | dict:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            dict: Dictionary of keyword arguments for cellpose.eval.
        """
        if self.mode == "default":
            return True
        if self.mode == "none":
            return False
        # Custom mode
        return {
            "normalize": self.normalize,
            "lowhigh": self.lowhigh,
            "sharpen": self.sharpen,
            "percentile": self.percentile,
            "tile_norm_blocksize": self.tile_norm_blocksize,
            "norm3D": self.norm3D,
        }


class AdvancedCellposeParameters(BaseModel):
    """Advanced Cellpose Model Parameters

    Attributes:
        normalization (NormalizationParameters, optional): Normalization parameters.
            The normalization is applied before running the Cellpose model for
            each channel independently.
        batch_size (int, optional): number of 256x256 patches to run simultaneously
            on the GPU (can make smaller or bigger depending on GPU memory usage).
            Defaults to 8.
        resample (bool, optional): run dynamics at original image size
            (will be slower but create more accurate boundaries).
            Defaults to True.
        invert (bool, optional): invert image pixel intensity before running network.
            Defaults to False.
        diameter (float or list of float, optional): diameters are
            used to rescale the image to 30 pix cell diameter.
        flow_threshold (float, optional): flow error threshold
            (all cells with errors below threshold are kept) (not used for 3D).
            Defaults to 0.4.
        cellprob_threshold (float, optional): all pixels with
            value above threshold kept for masks, decrease to find more and
            larger masks. Defaults to 0.0.
        flow3D_smooth (int, optional): if do_3D and flow3D_smooth>0, smooth flows
            with gaussian filter of this stddev. Defaults to 0.
        anisotropy (float, optional): for 3D segmentation, optional
            rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y).
            Defaults to None.
        stitch_threshold (float, optional): if stitch_threshold>0.0
            and not do_3D, masks are stitched in 3D to return volume segmentation.
            Defaults to 0.0.
        min_size (int, optional): all ROIs below this size,
            in pixels, will be discarded. Defaults to 15.
        max_size_fraction (float, optional): max_size_fraction
            (float, optional): Masks larger than max_size_fraction of total image size
            are removed. Default is 0.4.
        niter (int, optional): number of iterations for dynamics computation
            if None, it is set proportional to the diameter. Defaults to None.
        augment (bool, optional): tiles image with overlapping tiles and
            flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): fraction of overlap of tiles
            when computing flows. Defaults to 0.1.
        bsize (int, optional): block size for tiles, recommended to
            keep at 224, like in training. Defaults to 256.
        verbose (bool, optional): whether cellpose logs should be active.
            Defaults to False.
    """

    normalization: NormalizationParameters = NormalizationParameters()
    batch_size: int = 8
    resample: bool = True
    invert: bool = False
    diameter: Optional[float] = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    flow3D_smooth: int = 0
    anisotropy: Optional[float] = None
    stitch_threshold: float = 0.0
    min_size: int = 15
    max_size_fraction: float = 0.4
    niter: Optional[int] = None
    augment: bool = False
    tile_overlap: float = 0.1
    bsize: int = 256
    verbose: bool = False

    def to_eval_kwargs(self) -> dict:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            dict: Dictionary of keyword arguments for cellpose.eval.
        """
        kwargs = {
            "batch_size": self.batch_size,
            "resample": self.resample,
            "invert": self.invert,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "flow3D_smooth": self.flow3D_smooth,
            "anisotropy": self.anisotropy,
            "stitch_threshold": self.stitch_threshold,
            "min_size": self.min_size,
            "max_size_fraction": self.max_size_fraction,
            "niter": self.niter,
            "augment": self.augment,
            "tile_overlap": self.tile_overlap,
            "bsize": self.bsize,
            # "interp": self.interp, TODO: interp only shown in the docstring
            "normalize": self.normalization.to_eval_kwargs(),
        }
        return kwargs


class CellposeChannels(BaseModel):
    """Cellpose channels configuration.

    Args:
        This model is used to select a channel by label, wavelength ID, or index.

    Args:
        identifiers (str): Unique identifier for the channel.
            This can be a channel label, wavelength ID, or index.
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer). At least one and at most three
            identifiers must be provided.

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifiers: list[str] = Field(default_factory=list, min_length=1, max_length=3)

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]
