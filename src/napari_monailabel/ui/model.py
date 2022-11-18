__all__ = ["ModelView"]

from qtpy.QtWidgets import *

from ..client import MonaiLabelInterface as MonaiLabel
from .interactor import *


class ModelView(QWidget):
    def __init__(self, name, type, model, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.type = type
        self.model = model
        self.trainer = trainer

        if self.type is None:
            self.type = MonaiLabel().client.get_model_type(self.name)

        self.interactor = (
            PointInteractor(self.name)
            if self.type == "deepedit"
            else BrushInteractor(self.name)
        )

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.interactor)
        self.setLayout(layout)
