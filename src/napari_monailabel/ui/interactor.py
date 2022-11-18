__all__ = ["BrushInteractor", "PointInteractor"]

from qtpy.QtWidgets import *

from .qitemmodels import *
from .qitemviews import *


class BrushInteractor(QWidget):
    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_view = LabelsView(model)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.labels_view)
        self.setLayout(layout)


class PointInteractor(BrushInteractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LabelsView(QWidget):
    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_view = TableView()
        self.labels_model = LabelsModel(model)
        self.labels_view.setModel(self.labels_model)
        self.labels_view.resizeColumnsToContents()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.labels_view)
        self.setLayout(layout)
