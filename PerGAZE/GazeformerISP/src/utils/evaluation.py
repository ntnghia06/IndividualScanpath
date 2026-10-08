from collections import defaultdict
import numpy as np
from tqdm import tqdm
from utils.evaltools.scanmatch import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity
from geometry import SCANPATH_SIZE
SCREEN = SCANPATH_SIZE

METRIC_PROTOCOL = "original_mm_retrieval_duration_v2"


def padded_scanpath(path):
    path = np.asarray(path, dtype=float).reshape(-1, 3)
    if len(path) < 3:
        path = np.concatenate([path, np.tile([1., 1., .001], (3 - len(path), 1))])
    return path


class Metrics:
    def __init__(self):
        self.scanmatches = [ScanMatch(Xres=SCREEN[1], Yres=SCREEN[0], Xbin=16, Ybin=12,
            Offset=(0, 0), Threshold=3.5, TempBin=ms) for ms in (0, 50)]

    def scanmatch(self, target, prediction):
        if not len(prediction):
            return (0., 0.)
        target, prediction = target.astype(float).copy(), prediction.astype(float).copy()
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            return (float("nan"), float("nan"))
        target[:, 2] *= 1000
        prediction[:, 2] *= 1000
        result = []
        for matcher in self.scanmatches:
            a = matcher.fixationToSequence(target).astype(np.int32)
            b = matcher.fixationToSequence(prediction).astype(np.int32)
            result.append(float(matcher.match(a, b)[0]) if len(a) and len(b) else 0.)
        return tuple(result)

    def multimatch(self, target, prediction):
        import multimatch_gaze
        dtype = [("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")]
        return np.asarray(multimatch_gaze.docomparison(
            np.array([tuple(r) for r in padded_scanpath(target)], dtype=dtype),
            np.array([tuple(r) for r in padded_scanpath(prediction)], dtype=dtype),
            screensize=[SCREEN[1], SCREEN[0]]), dtype=float)

    def reward(self, target, prediction):
        # Original pairs_eval rejects a trial when MultiMatch contains NaN.
        if not np.isfinite(self.multimatch(target, prediction)).all():
            return float("nan")
        a, b = self.scanmatch(target, prediction)
        if not np.isfinite([a, b]).all():
            return float("nan")
        return 2 * a * b / (a + b) if a + b else 0.

    def retrieval_score(self, target, prediction):
        # Original evaluator computes ScanMatch independently of MultiMatch NaN.
        return self.scanmatch(target, prediction)[1]

    def pair(self, target, prediction):
        a, b = self.scanmatch(target, prediction)
        stimulus = np.zeros(SCREEN, dtype=np.uint8)
        scores = self.multimatch(target, prediction)
        result = {"ScanMatch_without_duration": a, "ScanMatch_with_duration": b,
            "SED": float(string_edit_distance(stimulus, target, prediction)),
            "STDE": float(scaled_time_delay_embedding_similarity(target[:, :2], prediction[:, :2], stimulus)) if len(prediction) else 0.,
            "MultiMatch_padded_target": float(len(target) < 3),
            "MultiMatch_padded_prediction": float(len(prediction) < 3),
            "MultiMatch_padded_pair": float(min(len(target), len(prediction)) < 3),
            "MultiMatch_invalid_pair": float(not np.isfinite(scores).all())}
        result.update({"MultiMatch_" + name: float(value) if np.isfinite(scores).all() else None
                       for name, value in zip(("vector", "direction", "length", "position", "duration"), scores)})
        return {key: value if value is None or np.isfinite(value) else None for key, value in result.items()}


def retrieval(rows, targets, metrics):
    groups = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["name"], row["task"], row.get("qid") or "", row["repeat"])
        groups[key].append(row)
    details, ranked = [], []
    for key, queries in tqdm(sorted(groups.items()), desc="Subject retrieval", unit="group"):
        observers = sorted({row["subject_key"] for row in queries})
        candidates = {observer: {} for observer in observers}
        for row in queries:
            candidates[row["subject_key"]][row["record_index"]] = targets[row["record_index"]]
        matrix, ranks = [], []
        for query in queries:
            predicted = np.asarray(query["scanpath"], dtype=float).reshape(-1, 3)
            scores = []
            for observer in observers:
                values = [metrics.retrieval_score(gt, predicted) for gt in candidates[observer].values()]
                finite = [value for value in values if np.isfinite(value)]
                scores.append(float(max(finite)) if finite else None)
            matrix.append(scores)
            # Original p2g represents invalid similarities as -1 and skips only all-invalid rows.
            numeric = np.array([-1. if value is None else value for value in scores])
            if (numeric == -1).all():
                ranks.append(None)
                continue
            ordering = np.argsort(numeric)[::-1]
            rank = int(np.where(ordering == observers.index(query["subject_key"]))[0][0])
            ranks.append(rank + 1)
            ranked.append((key[0], rank, len(observers)))
        details.append({"condition": key[0], "name": key[1], "task": key[2], "qid": key[3] or None,
            "repeat": key[4], "candidate_subjects": observers, "candidate_count": len(observers),
            "query_subjects": [q["subject_key"] for q in queries],
            "query_record_indices": [q["record_index"] for q in queries],
            "ScanMatch_with_duration_matrix": matrix, "ranks_1based": ranks})

    def aggregate(condition=None):
        queries = [r for r in ranked if condition is None or r[0] == condition]
        ranks = np.array([r[1] for r in queries])
        result = {"queries": len(queries), "single_candidate_queries": sum(r[2] == 1 for r in queries),
            "candidate_counts": sorted({r[2] for r in queries})}
        result.update({f"R@{k}": float(100 * np.mean(ranks < k)) if len(ranks) else None for k in (1, 3, 5, 10)})
        result.update(MRR=float(np.mean(1 / (ranks + 1))) if len(ranks) else None,
            median_rank=float(np.floor(np.median(ranks)) + 1) if len(ranks) else None,
            mean_rank=float(np.mean(ranks) + 1) if len(ranks) else None)
        result.update(pr1=result["R@1"], pr3=result["R@3"], pr5=result["R@5"], pmrr=result["MRR"],
            rsum=sum(result[f"R@{k}"] for k in (1, 3, 5)) if len(ranks) else None)
        return result
    return {"metric": "ScanMatch_with_duration", "direction": "prediction_to_ground_truth",
        "duplicate_gt_policy": "max similarity across records of the same observer",
        "invalid_queries": len(rows) - len(ranked), "overall": aggregate(),
        "by_condition": {c: aggregate(c) for c in ("present", "absent", "vqa")}, "groups": details}


def summarize(rows, targets=None, metrics=None):
    def aggregate(samples):
        values = defaultdict(list)
        for row in samples:
            for key, value in row["metrics"].items():
                values[key]  # Retain metric names even if every comparison is invalid.
                if value is not None and np.isfinite(value):
                    values[key].append(value)
        return {"samples": len(samples), "metrics": {key: {"mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v)) if v else None, "valid_count": len(v)} for key, v in values.items()},
            "multimatch_padding": {"pairs": int(sum(r["metrics"].get("MultiMatch_padded_pair", 0) for r in samples)),
                "targets": int(sum(r["metrics"].get("MultiMatch_padded_target", 0) for r in samples)),
                "predictions": int(sum(r["metrics"].get("MultiMatch_padded_prediction", 0) for r in samples)),
                "invalid_pairs": int(sum(r["metrics"].get("MultiMatch_invalid_pair", 0) for r in samples))}}
    result = {"metric_protocol": METRIC_PROTOCOL, "overall": aggregate(rows), "by_condition": {
        c: aggregate([r for r in rows if r["condition"] == c]) for c in ("present", "absent", "vqa")},
        "by_subject": {subject: aggregate([r for r in rows if r["subject_key"] == subject])
                       for subject in sorted({r["subject_key"] for r in rows})}}
    if targets is not None:
        result["retrieval"] = retrieval(rows, targets, metrics or Metrics())
    return result
