"""Pydantic models for advanced iterator configuration."""

from typing import Annotated, Literal

from fractal_tasks_utils.segmentation import IteratorConfig, MaskingConfig
from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "IteratorConfig",
    "MaskingConfig",
]


class DefaultNorm(BaseModel):
    """Default Cellpose normalization."""

    mode: Literal["default"] = "default"
    """
    Use Cellpose default normalization (0-1 rescaling and 1%-99% percentile clipping).
    All other parameters are ignored.
    """

    def to_eval_kwargs(self) -> bool:
        """Convert to dictionary of keyword arguments for cellpose.eval.

        Returns:
            bool: True indicating default normalization.
        """
        return True


class NoNorm(BaseModel):
    """Disable Cellpose normalization."""

    mode: Literal["none"] = "none"
    """
    Do not apply any normalization, and ignore all other parameters.
    """

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
    """

    mode: Literal["custom"] = "custom"
    """Use custom normalization with full control over parameters."""
    normalize: bool = True
    """Whether to perform normalization."""
    norm3D: bool = False
    """
    Whether to normalize in 3D. If True, the entire 3D stack will be normalized
    per channel. If False, normalization is applied per Z-slice.
    """
    invert: bool = False
    """Whether to invert the image. Useful if cells are dark instead of bright."""
    use_lowhigh: bool = False
    """
    Whether to use lowhigh normalization. If False, percentile normalization will be
    used instead.
    """
    lowhigh: tuple[float, float] = (0.0, 1000.0)
    """
    The lower and upper bounds for normalization. Incompatible with smoothing and
    sharpening.
    """
    percentile: tuple[float, float] = (1.0, 99.0)
    """
    The lower and upper percentiles for normalization. Each value should be between
    0 and 100. If use_lowhigh is True, this parameter is ignored.
    """
    sharpen_radius: float = 0.0
    """The radius for sharpening the image."""
    smooth_radius: float = 0.0
    """The radius for smoothing the image."""
    tile_norm_blocksize: int = 0
    """The block size for tile-based normalization."""
    tile_norm_smooth3D: int = 1
    """The smoothness factor for tile-based normalization in 3D."""
    axis: int = -1
    """The channel axis to loop over for normalization."""

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
    """Create Masking ROI Table Configuration."""

    mode: Literal["Create Masking ROI Table"] = "Create Masking ROI Table"
    """Mode to create masking ROI table."""
    table_name: str = "{label_name}_masking_ROI_table"
    """
    Name of the masking ROI table to be created. {label_name} is the name of the
    label image used for segmentation.
    """

    def get_table_name(self, label_name: str) -> str:
        """Get the actual table name by replacing placeholder.

        Args:
            label_name (str): Name of the label image used for segmentation.

        Returns:
            str: Actual name of the masking ROI table.
        """
        return self.table_name.format(label_name=label_name)


class SkipCreateMaskingRoiTable(BaseModel):
    """Skip Creating Masking ROI Table Configuration."""

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"
    """Mode to skip creating masking ROI table."""


AnyCreateRoiTableModel = Annotated[
    CreateMaskingRoiTable | SkipCreateMaskingRoiTable,
    Field(discriminator="mode"),
]


class AdvancedCellposeParameters(BaseModel):
    """Advanced Cellpose Model Parameters."""

    normalization: AnyNormModel = DefaultNorm()
    """
    Normalization model to use. Options are DefaultNorm (use Cellpose default
    normalization), NoNorm (disable normalization), or CustomNorm (full control over
    normalization parameters).
    """
    batch_size: int = 8
    """
    Number of 256x256 patches to run simultaneously on the GPU (can make smaller or
    bigger depending on GPU memory usage).
    """
    resample: bool = True
    """
    Run dynamics at original image size (will be slower but create more accurate
    boundaries).
    """
    invert: bool = False
    """Invert image pixel intensity before running network."""
    diameter: float | None = None
    """Diameter used to rescale the image to 30 pix cell diameter."""
    flow_threshold: float = 0.4
    """
    Flow error threshold (all cells with errors below threshold are kept)
    (not used for 3D).
    """
    cellprob_threshold: float = 0.0
    """
    All pixels with value above threshold kept for masks, decrease to find more and
    larger masks.
    """
    flow3D_smooth: int = 0
    """
    If do_3D and flow3D_smooth>0, smooth flows with gaussian filter of this stddev.
    """
    anisotropy: float | None = None
    """
    For 3D segmentation, rescaling factor (e.g. set to 2.0 if Z is sampled half as
    dense as X or Y).
    """
    do_3D: bool = True
    """
    Whether to perform 3D segmentation. If set to False for a 3D image, the image will
    be segmented by XY planes independently and then stitched together in 3D using the
    stitch_threshold parameter. If set to True for a 2D image, this will be ignored.
    """
    stitch_threshold: float = 0.0
    """
    If stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume
    segmentation.
    """
    min_size: int = 15
    """All ROIs below this size, in pixels, will be discarded."""
    max_size_fraction: float = 0.4
    """Masks larger than max_size_fraction of total image size are removed."""
    niter: int | None = None
    """
    Number of iterations for dynamics computation. If None, it is set proportional to
    the diameter.
    """
    augment: bool = False
    """Tiles image with overlapping tiles and flips overlapped regions to augment."""
    tile_overlap: float = 0.1
    """Fraction of overlap of tiles when computing flows."""
    bsize: int = 256
    """Block size for tiles, recommended to keep at 224, like in training."""
    verbose: bool = False
    """Whether cellpose logs should be active."""

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
            "do_3D": self.do_3D,
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

    This model is used to select a channel by label, wavelength ID, or index.
    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    """
    Specifies how to interpret the identifiers. Can be "label", "wavelength_id", or
    "index" (must be an integer).
    """
    identifiers: list[str] = Field(min_length=1, max_length=3)
    """
    Unique identifiers for the channels. This can be channel labels, wavelength IDs, or
    indices, depending on the mode.
    At least one and at most three identifiers must be provided.
    """
    skip_if_missing: bool = False
    """
    If True and the specified channel(s) are not found in the image,
    the segmentation will be skipped instead of raising an error. Defaults to False.
    """

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
