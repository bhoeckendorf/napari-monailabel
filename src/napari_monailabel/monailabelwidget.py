from typing import Any, Optional

import napari
import numpy as np
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from .monailabelclient import MonaiLabelClient


class InfoModel(QAbstractTableModel):

    def __init__(self, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self._col_headers = ("Value",)
        self._row_headers = ("Name", "Version")

    def columnCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return 1

    def rowCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return len(self._row_headers)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._col_headers[section]
        else:
            return self._row_headers[section]

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        attr = self._row_headers[index.row()].lower()
        return getattr(self.client, attr)


class LabelsModel(QAbstractTableModel):

    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels = labels
        self._keys = tuple(self._labels.keys())

    def columnCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return 2

    def rowCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return len(self._labels)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        key = self._keys[index.row()]
        if index.column() == 0:
            return self._labels[key]
        elif index.column() == 1:
            return key


class BrushInteractor(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointInteractor(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModelCtrl(QWidget):
    inference_requested = Signal(str)

    def __init__(self, name, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._model_type = str(model["type"]).lower()
        self.labels_view = QTableView()
        self.labels_model = LabelsModel(model["labels"])
        self.labels_view.setModel(self.labels_model)

        self.interactor = PointInteractor() if self._model_type == "deepedit" else BrushInteractor()

        infer_btn = QPushButton("Inference")
        infer_btn.clicked.connect(self._on_infer)

        layout = QGridLayout()
        layout.addWidget(QLabel(self._model_type))
        layout.addWidget(self.labels_view)
        layout.addWidget(self.interactor)
        layout.addWidget(infer_btn)
        self.setLayout(layout)

    def _on_infer(self):
        self.inference_requested.emit(self._name)


class DatastoreModel(QAbstractTableModel):

    def __init__(self, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self._row_items = tuple(self.client.datastore["objects"].keys())

    def columnCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return 2

    def rowCount(self, parent: Optional[QModelIndex] = QModelIndex()) -> int:
        return len(self._row_items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        key = self._row_items[index.row()]
        if index.column() == 0:
            return key
        elif index.column() == 1:
            return len(self.client.datastore["objects"][key]["labels"])


class ServerCtrl(QWidget):
    def __init__(self, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

        self.url_ctrl = QLineEdit()
        connect_btn = QPushButton("...")
        connect_btn.clicked.connect(self._on_connect)

        self._on_connect()

        self.info_view = QTableView()

        self.info_model = InfoModel(self.client)
        self.info_view.setModel(self.info_model)

        layout = QGridLayout()
        layout.addWidget(QLabel("URL"), 0, 0)
        layout.addWidget(self.url_ctrl, 0, 1)
        layout.addWidget(connect_btn, 0, 2)
        layout.addWidget(self.info_view, 1, 0, 1, 3)
        self.setLayout(layout)

    def _on_connect(self):
        url = self.url_ctrl.text().strip()
        self.client.connect(url)


class ActiveLearningCtrl(QWidget):
    image_requested = Signal(str)

    def __init__(self, client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

        self.progress_view = QProgressBar()
        self.strategies_model = QStringListModel()
        self.strategies_model.setStringList(["First", "Random"])
        self.strategy_ctrl = QComboBox()
        self.strategy_ctrl.setModel(self.strategies_model)

        next_btn = QPushButton("Next sample")
        submit_btn = QPushButton("Submit label")

        next_btn.clicked.connect(self._on_next)

        layout = QGridLayout()
        layout.addWidget(QLabel("Progress"), 0, 0)
        layout.addWidget(self.progress_view, 0, 1)
        layout.addWidget(QLabel("Strategy"), 1, 0)
        layout.addWidget(self.strategy_ctrl, 1, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(next_btn)
        btn_layout.addWidget(submit_btn)
        layout.addLayout(btn_layout, 2, 0, 1, 2)
        self.setLayout(layout)

    def _on_next(self):
        sample = self.client.get_active_learning_sample()
        self.image_requested.emit(sample["id"])


class DatastoreView(QWidget):
    image_requested = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = QTableView()
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

    def setModel(self, model):
        self.view.setModel(model)
        sel = self.view.selectionModel()
        sel.selectionChanged.connect(self._onSelectionChanged)

    def model(self):
        return self.view.model()

    def _onSelectionChanged(self, sel, desel):
        names = set(i.row() for i in self.view.selectedIndexes())
        names = [self.view.model().index(r, 0).data(Qt.DisplayRole) for r in names]
        self.image_requested.emit(names[0])


class MonaiLabelWidget(QWidget):

    def __init__(self, viewer: napari.Viewer, *args, client: Optional[MonaiLabelClient] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.client = client if client is not None else MonaiLabelClient()

        self.server_ctrl = ServerCtrl(self.client)
        server_box = self._wrap_in_groupbox(self.server_ctrl, "Server")

        self.active_learning_ctrl = ActiveLearningCtrl(self.client)
        active_learning_box = self._wrap_in_groupbox(self.active_learning_ctrl, "Active learning")

        tab_widget = QTabWidget()
        for name, model in self.client.models.items():
            ctrl = ModelCtrl(name, model)
            ctrl.inference_requested.connect(self.infer)
            tab_widget.addTab(ctrl, name)

        self.datastore_view = DatastoreView()
        tab_widget.addTab(self.datastore_view, "Data")

        self.datastore_model = DatastoreModel(self.client)
        self.datastore_view.setModel(self.datastore_model)

        self.active_learning_ctrl.image_requested.connect(self.load_image)
        self.datastore_view.image_requested.connect(self.load_image)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(server_box, 0, 0)
        layout.addWidget(active_learning_box, 1, 0)
        layout.addWidget(tab_widget, 2, 0)
        self.setLayout(layout)

    @staticmethod
    def _wrap_in_groupbox(widget: QWidget, title: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        box.setLayout(layout)
        return box

    @Slot(str)
    def load_image(self, name: str):
        img, spc = self.client.get_image(name)
        self.viewer.add_image(img, name=name, scale=spc / np.min(spc))

    @Slot(str)
    def infer(self, model_name: str):
        image_name = None
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, napari.layers.Image):
                image_name = layer.name
                break
        if image_name is None:
            raise ValueError()
        msk, spc = self.client.infer(model_name.lower(), image_name)
        self.viewer.add_labels(msk, name=f"{image_name}_infer", scale=spc / np.min(spc))
