"""Pydantic models for advanced iterator configuration."""

from typing import Annotated, Literal

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field, field_validator


class MaskingConfiguration(BaseModel):
    """Masking configuration.

    Attributes:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str | None): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: str | None = None


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration.

    Attributes:
        masking (MaskingConfiguration | None): If set, the segmentation will be
            performed only within the confines of the specified mask. A mask can be
            specified either by a label image or a Masking ROI table.
        roi_table (str | None): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    masking: MaskingConfiguration | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")


class DefaultNorm(BaseModel):
    """Default Cellpose normalization.

    Attributes:
        mode (Literal["default"]): Use Cellpose default normalization (0-1 rescaling and
            1%-99% percentile clipping). All other parameters are ignored.
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

    Attributes:
        mode (Literal["none"]): Do not apply any normalization, and ignore all other
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

    Attributes:
        mode (Literal["custom"]): Use custom normalization with full control over
            parameters.
        normalize (bool): Whether to perform normalization. Defaults to True.
        norm3D (bool): Whether to normalize in 3D. If True, the entire 3D
            stack will be normalized per channel. If False, normalization is applied
            per Z-slice. Defaults to False.
        invert (bool): Whether to invert the image. Useful if cells are dark
            instead of bright. Defaults to False.
        use_lowhigh (bool): Whether to use lowhigh normalization. If False,
            percentile normalization will be used instead. Defaults to False.
        lowhigh (tuple[float, float]): The lower and upper bounds for
            normalization. Incompatible with smoothing and sharpening.
            Defaults to (0.0, 1000.0).
        percentile (tuple[float, float]): The lower and upper percentiles for
            normalization. Each value should be between 0 and 100. If use_lowhigh
            is True, this parameter is ignored. Defaults to (1.0, 99.0).
        sharpen_radius (float): The radius for sharpening the image. Defaults to 0.0.
        smooth_radius (float): The radius for smoothing the image. Defaults to 0.0.
        tile_norm_blocksize (int): The block size for tile-based normalization.
            Defaults to 0.
        tile_norm_smooth3D (int): The smoothness factor for tile-based normalization
            in 3D. Defaults to 1.
        axis (int): The channel axis to loop over for normalization. Defaults to -1.
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
    """Create Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Create Masking ROI Table"]): Mode to create masking ROI table.
        table_name (str): Name of the masking ROI table to be created.
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
    """Skip Creating Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Skip Creating Masking ROI Table"]): Mode to skip creating
            masking ROI table.
    """

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"


AnyCreateRoiTableModel = Annotated[
    CreateMaskingRoiTable | SkipCreateMaskingRoiTable,
    Field(discriminator="mode"),
]


class AdvancedCellposeParameters(BaseModel):
    """Advanced Cellpose Model Parameters.

    Attributes:
        normalization (AnyNormModel): Normalization model to use. Options are
            DefaultNorm (use Cellpose default normalization), NoNorm (disable
            normalization), or CustomNorm (full control over normalization
            parameters). Defaults to DefaultNorm().
        batch_size (int): Number of 256x256 patches to run simultaneously on the
            GPU (can make smaller or bigger depending on GPU memory usage).
            Defaults to 8.
        resample (bool): Run dynamics at original image size (will be slower but
            create more accurate boundaries). Defaults to True.
        invert (bool): Invert image pixel intensity before running network.
            Defaults to False.
        diameter (float | None): Diameter used to rescale the image to 30 pix cell
            diameter. Defaults to None.
        flow_threshold (float): Flow error threshold (all cells with errors below
            threshold are kept) (not used for 3D). Defaults to 0.4.
        cellprob_threshold (float): All pixels with value above threshold kept for
            masks, decrease to find more and larger masks. Defaults to 0.0.
        flow3D_smooth (int): If do_3D and flow3D_smooth>0, smooth flows with
            gaussian filter of this stddev. Defaults to 0.
        anisotropy (float | None): For 3D segmentation, rescaling factor (e.g. set
            to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        do_3D (bool): Whether to perform 3D segmentation. If set to False for a 3D
            image, the image will be segmented by XY planes independently and then
            stitched together in 3D using the stitch_threshold parameter. If set to
            True for a 2D image, this will be ignored.
        stitch_threshold (float): If stitch_threshold>0.0 and not do_3D, masks are
            stitched in 3D to return volume segmentation. Defaults to 0.0.
        min_size (int): All ROIs below this size, in pixels, will be discarded.
            Defaults to 15.
        max_size_fraction (float): Masks larger than max_size_fraction of total
            image size are removed. Defaults to 0.4.
        niter (int | None): Number of iterations for dynamics computation. If None,
            it is set proportional to the diameter. Defaults to None.
        augment (bool): Tiles image with overlapping tiles and flips overlapped
            regions to augment. Defaults to False.
        tile_overlap (float): Fraction of overlap of tiles when computing flows.
            Defaults to 0.1.
        bsize (int): Block size for tiles, recommended to keep at 224, like in
            training. Defaults to 256.
        verbose (bool): Whether cellpose logs should be active. Defaults to False.
    """

    normalization: AnyNormModel = DefaultNorm()
    batch_size: int = 8
    resample: bool = True
    invert: bool = False
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    flow3D_smooth: int = 0
    anisotropy: float | None = None
    do_3D: bool = True
    stitch_threshold: float = 0.0
    min_size: int = 15
    max_size_fraction: float = 0.4
    niter: int | None = None
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

    Attributes:
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer).
        identifiers (list[str]): Unique identifiers for the channels. This can
            be channel labels, wavelength IDs, or indices. At least one and at
            most three identifiers must be provided.
    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifiers: list[str] = Field(min_length=1, max_length=3)

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
