__all__ = ["InfoModel", "LabelsModel", "DatastoreModel"]

from typing import Any

from natsort import natsorted
from qtpy.QtCore import *
from qtpy.QtGui import *

from ..client import MonaiLabelInterface as MonaiLabel


class InfoModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = MonaiLabel().client
        self.column_headers = ("Value",)
        self.row_headers = ("Name", "Version")

    @Slot()
    def on_server_changed(self):
        self.beginResetModel()
        self.endResetModel()

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.column_headers)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        if self.client.is_connected():
            return len(self.row_headers)
        return 0

    def headerData(
            self, section: int, orientation: Qt.Orientation, role: int = ...
    ) -> Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.column_headers[section]
            else:
                return self.row_headers[section]
        return None

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if index.isValid() and role == Qt.DisplayRole:
            attr = self.row_headers[index.row()].lower()
            try:
                return getattr(self.client, attr)
            except KeyError:
                return None
        return None


class LabelsModel(QAbstractTableModel):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_headers = ("Color", "Name")
        self.row_headers = tuple()
        self.label_dict = {}
        self.label_names = tuple(self.label_dict.keys())
        self.model = model
        self.on_server_changed()

    @Slot()
    def on_server_changed(self):
        self.beginResetModel()
        client = MonaiLabel().client
        try:
            self.label_dict = client.get_model_labels(self.model)
        except KeyError:
            self.label_dict = {}
        self.label_names = tuple(self.label_dict.keys())
        self.endResetModel()

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.column_headers)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.label_dict)

    def headerData(
            self, section: int, orientation: Qt.Orientation, role: int = ...
    ) -> Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.column_headers[section]
            return self.label_dict[self.label_names[section]]
        return super().headerData(section, orientation, role)

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if index.isValid() and role == Qt.DisplayRole:
            label_name = self.label_names[index.row()]
            if index.column() == 0:
                return self.label_dict[label_name]
            elif index.column() == 1:
                return label_name
        return None


class DatastoreModel(QAbstractTableModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = MonaiLabel().client
        self.column_headers = ("Name", "Labels")
        self.objects = None
        self.row_items = tuple()

    @Slot()
    def on_server_changed(self):
        self.beginResetModel()
        try:
            self.objects = self.client.datastore["objects"]
            self.row_items = natsorted(self.objects.keys())
        except KeyError:
            self.row_items = tuple()
        self.endResetModel()

    def columnCount(self, parent: QModelIndex = ...) -> int:
        return len(self.column_headers)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.row_items)

    def headerData(
            self, section: int, orientation: Qt.Orientation, role: int = ...
    ) -> Any:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.column_headers[section]
        return super().headerData(section, orientation, role)

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if index.isValid() and role == Qt.DisplayRole:
            image_name = self.row_items[index.row()]
            if index.column() == 0:
                return image_name
            elif index.column() == 1:
                return ", ".join(natsorted(self.objects[image_name]["labels"].keys()))
        return None
