"""Pydantic models for advanced iterator configuration."""

from typing import Literal, Optional

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
        masking (Optional[MaskingIterator]): If configured, the segmentation
            will be only saved within the mask region.
        roi_table (Optional[str]): Name of a ROI table. If provided, the segmentation
            will be performed for each ROI in the specified ROI table.
    """

    masking: Optional[MaskingConfiguration] = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: Optional[str] = Field(default=None, title="Iterate Over ROIs")
