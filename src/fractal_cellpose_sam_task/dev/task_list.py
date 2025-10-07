"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Lorenzo Cerrone"


DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task"


INPUT_MODELS = [
    ("ngio", "images/_image.py", "ChannelSelectionModel"),
    (
        "fractal_cellpose_sam_task",
        "utils.py",
        "MaskingConfiguration",
    ),
    (
        "fractal_cellpose_sam_task",
        "utils.py",
        "IteratorConfiguration",
    ),
]

TASK_LIST = [
    
    ParallelTask(
        name="Cellpose SAM Segmentation",
        executable="cellpose_sam_segmentation_task.py",
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/cellpose_sam_segmentation_task.md",
    ),
    
    
    
]
