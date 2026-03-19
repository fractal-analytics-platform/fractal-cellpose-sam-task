"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Lorenzo Cerrone"


DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task"

TASK_LIST = [
    ParallelTask(
        name="Cellpose SAM Segmentation",
        executable="cellpose_sam_segmentation_task.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
        category="Segmentation",
        tags=[
            "Deep Learning",
            "Convolutional Neural Network",
            "Instance Segmentation",
            "2D",
            "3D",
            "Cellpose",
            "SAM",
        ],
        docs_info="file:docs_info/cellpose_sam_segmentation_task.md",
    ),
]
