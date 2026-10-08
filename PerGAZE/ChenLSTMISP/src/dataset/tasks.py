"""Original COCO object order, with a separate shared VQA branch."""
OBJECT_NAMES = ("bottle", "bowl", "car", "chair", "clock", "cup", "fork",
                "keyboard", "knife", "laptop", "microwave", "mouse", "oven",
                "potted plant", "sink", "stop sign", "toilet", "tv")
OBJECT_TO_INDEX = {name: index for index, name in enumerate(OBJECT_NAMES)}
HEAD_SCHEMA = "coco18_tp_ta_air_vqa_v1"


def task_head(row):
    if row["condition"] == "vqa":
        return -1
    if row["condition"] not in ("present", "absent"):
        raise ValueError(f"Unknown condition: {row['condition']}")
    name = " ".join(row["task"].lower().replace("_", " ").split())
    if name not in OBJECT_TO_INDEX:
        raise ValueError(f"Unknown TP/TA object task: {row['task']!r}")
    return OBJECT_TO_INDEX[name]
