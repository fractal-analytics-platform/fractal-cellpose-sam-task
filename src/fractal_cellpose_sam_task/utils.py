"""Pydantic models for advanced iterator configuration."""

from typing import Annotated, Literal, Optional

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field, field_validator


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


class DefaultNorm(BaseModel):
    """Default Cellpose normalization.

    Args:
        mode: Literal["default"]: Use Cellpose default normalization (0-1 rescaling and
            1%-99% percentile clipping). all other parameters are ignored.
    """

    mode: Literal["default"] = "default"

    def to_eval_kwargs(self) -> bool:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            bool: True indicating default normalization.
        """
        return True


class NoNorm(BaseModel):
    """Disable Cellpose normalization.

    Args:
        mode: Literal["none"]: Do not apply any normalization, and ignore all other
            parameters.
    """

    mode: Literal["none"] = "none"

    def to_eval_kwargs(self) -> bool:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            bool: False indicating no normalization.
        """
        return False


class CustomNorm(BaseModel):
    """Custom Cellpose normalization.

    The normalization is applied before running the Cellpose model for
    each channel independently.

    Args:
        mode: Literal["custom"]: Use custom normalization with full control over
            parameters.
        normalize (bool, optional): Whether to perform normalization. Defaults to True.
        norm3D (bool, optional): Whether to normalize in 3D. If True, the entire 3D
            stack will be normalized per channel. If False, normalization is applied
            per Z-slice. Defaults to False.
        invert (bool, optional): Whether to invert the image. Useful if cells are dark
            instead of bright. Defaults to False.
        use_lowhigh (bool, optional): Whether to use lowhigh normalization. If False,
            percentile normalization will be used instead. Defaults to False.
        lowhigh (tuple or ndarray, optional): The lower and upper bounds for
            normalization. Can be a tuple of two values (applied to all channels) or
            an array of shape (nchan, 2) for per-channel normalization. Incompatible
            with smoothing and sharpening. Defaults to None.
        percentile (tuple, optional): The lower and upper percentiles for normalization.
            If provided, it should be a tuple of two values. Each value should be
            between 0 and 100. If use_lowhigh is True, this parameter is ignored.
            Defaults to (1.0, 99.0).
        sharpen_radius (int, optional): The radius for sharpening the image. Defaults
            to 0.
        smooth_radius (int, optional): The radius for smoothing the image. Defaults
            to 0.
        tile_norm_blocksize (int, optional): The block size for tile-based
            normalization. Defaults to 0.
        tile_norm_smooth3D (int, optional): The smoothness factor for tile-based
            normalization in 3D. Defaults to 1.
        axis (int, optional): The channel axis to loop over for normalization.
            Defaults to -1.
    """

    mode: Literal["custom"] = "custom"
    normalize: bool = True
    norm3D: bool = False
    invert: bool = False
    use_lowhigh: bool = False
    lowhigh: tuple[float, float] = (0.0, 1000.0)
    percentile: tuple[float, float] = (1.0, 99.0)
    sharpen_radius: float = 0.0
    smooth_radius: float = 0.0
    tile_norm_blocksize: int = 0
    tile_norm_smooth3D: int = 1
    axis: int = -1

    def to_eval_kwargs(self) -> bool | dict:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            dict: Dictionary of keyword arguments for cellpose.eval.
        """
        return {
            "normalize": self.normalize,
            "norm3D": self.norm3D,
            "invert": self.invert,
            "lowhigh": self.lowhigh if self.use_lowhigh else None,
            "percentile": self.percentile,
            "sharpen_radius": self.sharpen_radius,
            "smooth_radius": self.smooth_radius,
            "tile_norm_blocksize": self.tile_norm_blocksize,
            "tile_norm_smooth3D": self.tile_norm_smooth3D,
            "axis": self.axis,
        }


AnyNormModel = Annotated[DefaultNorm | NoNorm | CustomNorm, Field(discriminator="mode")]


class CreateMaskingRoiTable(BaseModel):
    """Create Masking ROI Table Configuration

    Attributes:
        mode: Literal["Create Masking ROI Table"]: Mode to create masking ROI table.
        table_name: str: Name of the masking ROI table to be created.
            Defaults to "{label_name}_masking_ROI_table", where {label_name} is
            the name of the label image used for segmentation.

    """

    mode: Literal["Create Masking ROI Table"] = "Create Masking ROI Table"
    table_name: str = "{label_name}_masking_ROI_table"

    def get_table_name(self, label_name: str) -> str:
        """Get the actual table name by replacing placeholder.

        Args:
            label_name (str): Name of the label image used for segmentation.

        Returns:
            str: Actual name of the masking ROI table.
        """
        return self.table_name.format(label_name=label_name)


class SkipCreateMaskingRoiTable(BaseModel):
    """Skip Creating Masking ROI Table Configuration

    Attributes:
        mode: Literal["Skip Creating Masking ROI Table"]: Mode to skip creating masking
            ROI table
    """

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"


AnyCreateRoiTableModel = Annotated[
    CreateMaskingRoiTable | SkipCreateMaskingRoiTable,
    Field(discriminator="mode"),
]


class AdvancedCellposeParameters(BaseModel):
    """Advanced Cellpose Model Parameters

    Attributes:
        normalization (AnyNormModel, optional): Normalization model to use.
            Options are DefaultNorm (use Cellpose default normalization),
            NoNorm (disable normalization), or CustomNorm (full control over
            normalization parameters). Defaults to DefaultNorm().
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

    normalization: AnyNormModel = DefaultNorm()
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

    @field_validator("identifiers", mode="after")
    @classmethod
    def validate_identifiers(cls, value: list[str]) -> list[str]:
        """Validate identifiers are non-empty"""
        for identifier in value:
            if not identifier:
                raise ValueError("Identifiers must be non-empty strings.")
        return value

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]
