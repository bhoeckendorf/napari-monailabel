import datetime
from typing import Any, Optional

import napari
import numpy as np
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from ..client import MonaiLabelInterface as MonaiLabel
from .qitemmodels import *


class TableView(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)


class BrushInteractor(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointInteractor(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ServerCtrl(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url_ctrl = QLineEdit()
        self.connect_btn = QPushButton("Connect")
        self.info_view = TableView()

        self.info_model = InfoModel()
        self.info_view.setModel(self.info_model)

        MonaiLabel().server_changed.connect(self.info_model.on_server_changed)
        self.connect_btn.clicked.connect(self._on_connect)

        layout = QGridLayout()
        layout.addWidget(QLabel("URL"), 0, 0)
        layout.addWidget(self.url_ctrl, 0, 1)
        layout.addWidget(self.connect_btn, 0, 2)
        layout.addWidget(self.info_view, 1, 0, 1, 3)
        self.setLayout(layout)

    def _on_connect(self):
        url = self.url_ctrl.text().strip()
        # TODO: validate url, ideally in GUI
        MonaiLabel().set_server(url)


class ActiveLearningCtrl(QWidget):
    image_requested = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_view = QProgressBar()
        self.strategies_model = QStringListModel()
        self.strategy_ctrl = QComboBox()
        self.strategy_ctrl.setModel(self.strategies_model)

        next_btn = QPushButton("Next sample")
        submit_btn = QPushButton("Submit label")

        next_btn.clicked.connect(self._on_next)
        MonaiLabel().server_changed.connect(self.on_server_changed)

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

    @Slot()
    def on_server_changed(self):
        self.strategies_model.beginResetModel()
        try:
            strategies = sorted(MonaiLabel().client.info["strategies"].keys())
            stats = MonaiLabel().client.get_datastore_stats()
        except KeyError:
            strategies = tuple()
            stats = {"total": 0, "completed": 0}
        self.strategies_model.setStringList(strategies)
        self.strategies_model.endResetModel()
        if self.strategy_ctrl.count() > 0:
            self.strategy_ctrl.setCurrentIndex(0)
        self.progress_view.setMaximum(stats["total"])
        self.progress_view.setValue(stats["completed"])

    def _on_next(self):
        sample = MonaiLabel().client.get_active_learning_sample()
        self.image_requested.emit(sample["id"])


class DatastoreView(QWidget):
    image_requested = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_ctrl = QLineEdit()
        self.filter_ctrl.setPlaceholderText("Filter ...")
        self.view = TableView()
        self.view.setSortingEnabled(True)
        self.model = DatastoreModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)

        self.view.setModel(self.proxy_model)
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.filter_ctrl.textChanged.connect(self.proxy_model.setFilterWildcard)
        MonaiLabel().server_changed.connect(self.model.on_server_changed)
        self.model.modelReset.connect(self.view.resizeColumnsToContents)

        layout = QVBoxLayout()
        layout.addWidget(self.filter_ctrl)
        layout.addWidget(self.view)
        self.setLayout(layout)

    def _on_selection_changed(self, sel, desel):
        names = set(i.row() for i in self.view.selectedIndexes())
        names = [self.view.model().index(r, 0).data(Qt.DisplayRole) for r in names]
        self.image_requested.emit(names[0])


class LogsView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.num_lines_ctrl = QSpinBox()
        refresh_btn = QPushButton("Refresh")

        self.num_lines_ctrl.setMinimum(100)
        self.num_lines_ctrl.setMaximum(900)
        self.num_lines_ctrl.setSingleStep(100)
        self.num_lines_ctrl.setValue(300)
        refresh_btn.clicked.connect(self.fetch)

        layout = QGridLayout()
        layout.addWidget(QLabel("Nr of lines"), 0, 0)
        layout.addWidget(self.num_lines_ctrl, 0, 1)
        layout.addWidget(refresh_btn, 0, 2)
        layout.addWidget(self.view, 1, 0, 1, 3)
        self.setLayout(layout)

    @Slot()
    def fetch(self, num_lines: Optional[int] = None):
        if num_lines is None:
            num_lines = self.num_lines_ctrl.value()
        logs = MonaiLabel().client.get_logs(num_lines=num_lines)
        self.view.setPlainText(logs)


class ModelCtrl(QWidget):
    def __init__(self, name, model, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.model = model
        self.trainer = trainer
        self.labels_view = TableView()
        self.labels_model = LabelsModel(name)
        self.labels_view.setModel(self.labels_model)
        self.labels_view.resizeColumnsToContents()

        self.interactor = (
            PointInteractor()
            if MonaiLabel().client.get_model_type(name) == "deepedit"
            else BrushInteractor()
        )

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.labels_view)
        layout.addWidget(self.interactor)
        self.setLayout(layout)


class ModelsCtrl(QWidget):
    inference_requested = Signal(str)
    training_start_requested = Signal(str)
    training_stop_requested = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_selector = QComboBox()
        self.model_type_view = QLineEdit()
        self.model_type_view.setReadOnly(True)
        self.models_model = QStringListModel()
        self.model_selector.setModel(self.models_model)
        self.metric_view = QProgressBar()
        self.metric_view.setMaximum(100)
        self.train_progress_view = QProgressBar()
        self.train_eta_view = QLineEdit()
        self.train_eta_view.setReadOnly(True)
        self.infer_btn = QPushButton("Predict")
        self.train_btn = QPushButton("Start training")
        self.stacked_layout = QStackedLayout()
        self._num_stacked = 0

        self.model_selector.currentIndexChanged.connect(
            self.stacked_layout.setCurrentIndex
        )
        self.model_selector.currentIndexChanged.connect(self._on_model_selection_changed)
        self.infer_btn.clicked.connect(self._on_infer)
        self.train_btn.clicked.connect(self._on_train)
        MonaiLabel().server_changed.connect(self.on_server_changed)
        MonaiLabel().training_started.connect(self.on_training_started)
        MonaiLabel().training_stopped.connect(self.on_training_stopped)
        MonaiLabel().training_progress.connect(self.on_training_progress)
        self.training_start_requested.connect(MonaiLabel().client.training_start)
        self.training_stop_requested.connect(MonaiLabel().client.training_stop)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.infer_btn)
        btn_layout.addWidget(self.train_btn)

        self.train_progress_label = QLabel("Training progress")
        self.train_eta_label = QLabel("Estimated completion")
        layout = QGridLayout()
        layout.addWidget(QLabel("Model"), 0, 0)
        layout.addWidget(self.model_selector, 0, 1)
        layout.addWidget(QLabel("Model type"), 1, 0)
        layout.addWidget(self.model_type_view, 1, 1)
        layout.addWidget(QLabel("Accuracy"), 2, 0)
        layout.addWidget(self.metric_view, 2, 1)
        layout.addWidget(self.train_progress_label, 3, 0)
        layout.addWidget(self.train_progress_view, 3, 1)
        layout.addWidget(self.train_eta_label, 4, 0)
        layout.addWidget(self.train_eta_view, 4, 1)
        layout.addLayout(btn_layout, 5, 0, 1, 2)
        layout.addLayout(self.stacked_layout, 6, 0, 1, 2)
        self.setLayout(layout)

    def insert_item(self, item: QWidget, name: str, index: int = 0):
        self.stacked_layout.insertWidget(index, item)
        self.models_model.insertRow(index)
        self.models_model.setData(self.models_model.index(index), name)
        self._num_stacked += 1

    def remove_item(self, index: int = 0) -> QWidget:
        self.models_model.removeRow(index)
        item = self.stacked_layout.widget(index)
        self.stacked_layout.removeWidget(item)
        self._num_stacked -= 1
        return item

    @Slot()
    def on_server_changed(self):
        while self._num_stacked > 0:
            _ = self.remove_item()
        try:
            models = MonaiLabel().client.models
            trainers = MonaiLabel().client.trainers
        except KeyError:
            models = {}
            trainers = {}
        for name in reversed(models.keys()):
            item = ModelCtrl(name, models[name], trainers.get(name, None))
            self.insert_item(item, name)
        if self.model_selector.count() > 0:
            self.model_selector.setCurrentIndex(0)

    def _on_model_selection_changed(self, index: int):
        model_ctrl = self.stacked_layout.widget(index)
        self.model_type_view.setText(model_ctrl.name)
        self._set_trainable(model_ctrl.trainer is not None)
        stats = MonaiLabel().client.info["train_stats"]
        if model_ctrl.name in stats:
            stats = stats[model_ctrl.name]
            if "best_metric" in stats:
                self.metric_view.setValue(int(round(stats["best_metric"] * 100)))
                return
        self.metric_view.setValue(0)

    def _set_trainable(self, state: bool):
        for i in (
            self.train_btn,
            self.train_progress_view,
            self.train_eta_view,
            self.train_progress_label,
            self.train_eta_label
        ):
            i.setVisible(state)

    def _on_infer(self):
        model = self.model_selector.currentText()
        if model is not None and len(model) > 0:
            self.inference_requested.emit(model)

    def _on_train(self):
        if self.train_btn.text().startswith("Start"):
            model = self.model_selector.currentText()
            if model is not None and len(model) > 0:
                self.training_start_requested.emit(model)
        elif self.train_btn.text().startswith("Stop"):
            self.training_stop_requested.emit()

    @Slot()
    def on_training_started(self):
        # self.infer_btn.setEnabled(False)
        self.train_btn.setStyleSheet(
            r"QPushButton {background-color: darkRed; border: none;}"
        )  # 1px solid black
        self.train_btn.setText("Stop training")
        self.train_progress_view.setValue(0)
        self.train_eta_view.setText("pending ...")

    @Slot()
    def on_training_stopped(self):
        self.train_btn.setStyleSheet("")
        self.train_btn.setText("Start training")
        self.train_progress_view.setValue(0)
        self.train_eta_view.setText("")
        # self.infer_btn.setEnabled(True)

    @Slot(int, int, datetime.datetime, float)
    def on_training_progress(
        self, current_epoch: int, max_epochs: int, eta: datetime.datetime, metric: float
    ):
        self.train_progress_view.setMaximum(max_epochs - 1)
        self.train_progress_view.setValue(current_epoch)
        self.train_eta_view.setText(" ".join(eta.isoformat().split("T")).split(".")[0])
        self.metric_view.setValue(int(round(metric * 100)))


class MonaiLabelWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self._client_interface = MonaiLabel()
        self._image_layer = None
        self._labels_layer = None
        self._prediction_layer = None

        self.server_ctrl = ServerCtrl()
        self.active_learning_ctrl = ActiveLearningCtrl()
        self.models_ctrl = ModelsCtrl()
        self.datastore_view = DatastoreView()
        self.logs_view = LogsView()

        self.active_learning_ctrl.image_requested.connect(self.load_image)
        self.models_ctrl.inference_requested.connect(self.infer)
        self.datastore_view.image_requested.connect(self.load_image)

        tab_widget = QTabWidget()
        tab_widget.addTab(self.datastore_view, "Data set")
        tab_widget.addTab(self.logs_view, "Server logs")

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_wrap_in_groupbox(self.server_ctrl, "Server"), 0, 0)
        layout.addWidget(
            _wrap_in_groupbox(self.active_learning_ctrl, "Active learning"), 1, 0
        )
        layout.addWidget(_wrap_in_groupbox(self.models_ctrl, "Models"), 2, 0)
        layout.addWidget(tab_widget, 3, 0)
        self.setLayout(layout)

    @Slot(str)
    def load_image(self, name: str, load_available_labels: bool = True):
        if self._image_layer is not None and self._image_layer.name == name:
            return

        client = MonaiLabel().client
        img, spc = client.get_image(name)
        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(
                img, name=name, scale=spc / np.min(spc)
            )
        else:
            self._image_layer.data = img
            self._image_layer.name = name
            self._image_layer.scale = spc / np.min(spc)
            self._image_layer.refresh()

        if load_available_labels:
            labels = client.get_available_labels(name)
            tag = "final"
            if tag in labels:
                msk, spc = client.get_label(name, tag)
                if self._labels_layer is None:
                    self._labels_layer = self.viewer.add_labels(
                        msk, name=f"{name}_labels_{tag}", scale=spc / np.min(spc)
                    )
                else:
                    self._labels_layer.data = msk
                    self._labels_layer.name = f"{name}_labels_{tag}"
                    self._labels_layer.scale = spc / np.min(spc)
                    self._labels_layer.refresh()
            elif self._labels_layer is not None:
                self._labels_layer.data = np.zeros_like(
                    self._image_layer.data, dtype=np.uint8
                )
                self._labels_layer.scale = spc / np.min(spc)
                self._labels_layer.name = "labels"
                self._labels_layer.refresh()
        elif self._labels_layer is not None:
            self._labels_layer.data = np.zeros_like(
                self._image_layer.data, dtype=np.uint8
            )
            self._labels_layer.scale = spc / np.min(spc)
            self._labels_layer.name = "labels"
            self._labels_layer.refresh()

        if self._prediction_layer is not None:
            self._prediction_layer.data = np.zeros_like(
                self._image_layer.data, dtype=np.uint8
            )
            self._prediction_layer.scale = spc / np.min(spc)
            self._prediction_layer.name = "prediction"
            self._prediction_layer.refresh()

    @Slot(str)
    def infer(self, model: str):
        if self._image_layer is None:
            return
        image = self._image_layer.name
        msk, spc = MonaiLabel().client.infer(model.lower(), image)
        if self._prediction_layer is None:
            self._prediction_layer = self.viewer.add_labels(
                msk, name=f"{image}_prediction", scale=spc / np.min(spc)
            )
        else:
            self._prediction_layer.data = msk
            self._prediction_layer.name = f"{image}_prediction"
            self._prediction_layer.scale = spc / np.min(spc)
            self._prediction_layer.refresh()


def _wrap_in_groupbox(widget: QWidget, title: str) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    box.setLayout(layout)
    return box
