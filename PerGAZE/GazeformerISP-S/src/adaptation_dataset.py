"""Reuse the original loader with separate TA, TP and VQA observers."""
from dataset.text import GazeformerPerGAZE
from adaptation import subject_key


class SubjectAdaptationDataset(GazeformerPerGAZE):
    def observer_key(self, row):
        return subject_key(row)
