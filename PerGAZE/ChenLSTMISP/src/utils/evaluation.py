from collections import defaultdict
import numpy as np

from utils.evaltools.scanmatch import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity


class Metrics:
    def __init__(self):
        self.scanmatches = [ScanMatch(Xres=320, Yres=240, Xbin=16, Ybin=12, Offset=(0, 0),
                                     Threshold=3.5, TempBin=bin_ms) for bin_ms in (0, 50)]

    def scanmatch(self, target, prediction):
        if len(prediction) == 0:
            return (0., 0.)
        target, prediction = target.astype(float).copy(), prediction.astype(float).copy()
        target[:, 2] *= 1000
        prediction[:, 2] *= 1000
        result = []
        for i, matcher in enumerate(self.scanmatches):
            # Random/untrained log-normal heads can produce hours-long fixations.
            # Treat these predictions as invalid for the temporal score instead
            # of expanding them into millions of 50 ms symbols. Real PerGAZE
            # fixation durations are all below 3 s; the 10 s bound is generous.
            if i == 1 and ((prediction[:, 2] > 10000).any()
                           or np.rint(prediction[:, 2] / 50).sum() > 8192):
                result.append(0.)
                continue
            a = matcher.fixationToSequence(target).astype(np.int32)
            b = matcher.fixationToSequence(prediction).astype(np.int32)
            result.append(float(matcher.match(a, b)[0]) if len(a) and len(b) else 0.)
        return tuple(result)

    def reward(self, target, prediction):
        a, b = self.scanmatch(target, prediction)
        return 2 * a * b / (a + b) if a + b > 0 else 0.

    def pair(self, target, prediction):
        a, b = self.scanmatch(target, prediction)
        stimulus = np.zeros((240, 320), dtype=np.uint8)
        metrics = {"ScanMatch_without_duration": a, "ScanMatch_with_duration": b,
                   "duration_outlier": float(bool(len(prediction)) and (
                       (prediction[:, 2] > 10).any() or np.rint(prediction[:, 2] * 20).sum() > 8192)),
                   "SED": float(string_edit_distance(stimulus, target, prediction)),
                   "STDE": float(scaled_time_delay_embedding_similarity(target[:, :2], prediction[:, :2], stimulus))
                           if len(prediction) else 0.}
        # Do not pad short scanpaths with invented fixations to obtain MultiMatch.
        try:
            import multimatch_gaze
        except ImportError:
            return metrics
        if min(len(target), len(prediction)) >= 3:
            dtype = [("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")]
            scores = multimatch_gaze.docomparison(
                np.array([tuple(row) for row in target], dtype=dtype),
                np.array([tuple(row) for row in prediction], dtype=dtype), screensize=[320, 240])
            metrics.update({"MultiMatch_" + name: float(value) for name, value in
                            zip(("vector", "direction", "length", "position", "duration"), scores)})
        return metrics


def summarize(rows):
    def aggregate(samples):
        values = defaultdict(list)
        for row in samples:
            for key, value in row["metrics"].items():
                if np.isfinite(value):
                    values[key].append(value)
        return {"samples": len(samples), "metrics": {k: {"mean": float(np.mean(v)),
                "std": float(np.std(v)), "valid_count": len(v)} for k, v in values.items()}}
    return {"overall": aggregate(rows), "by_condition": {
            condition: aggregate([r for r in rows if r["condition"] == condition])
            for condition in ("present", "absent", "vqa")}, "by_subject": {
            subject: aggregate([r for r in rows if r["subject_key"] == subject])
            for subject in sorted({r["subject_key"] for r in rows})}}
