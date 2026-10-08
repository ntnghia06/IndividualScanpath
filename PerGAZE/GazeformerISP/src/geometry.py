"""Fixed Gazeformer geometry; all tuples are (height, width)."""
IMAGE_SIZE = (768, 1024)
SCANPATH_SIZE = (352, 512)
FEATURE_GRID = (24, 32)
ACTION_GRID = FEATURE_GRID
ACTION_COUNT = ACTION_GRID[0] * ACTION_GRID[1] + 1
GEOMETRY = {"version": 3, "image_size": list(IMAGE_SIZE),
            "scanpath_size": list(SCANPATH_SIZE), "feature_grid": list(FEATURE_GRID),
            "action_grid": list(ACTION_GRID)}
