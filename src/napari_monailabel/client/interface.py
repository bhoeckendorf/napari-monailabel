import datetime
import re
import time
from copy import deepcopy

import numpy as np
from qtpy.QtCore import QObject, QThread, Signal, Slot

from .client import MonaiLabelClient


class MonaiLabelInterface(QObject):
    server_changed = Signal()
    training_started = Signal()
    training_stopped = Signal()
    training_progress = Signal(int, int, datetime.datetime, float)

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MonaiLabelInterface, cls).__new__(cls)
            super(QObject, cls.instance).__init__()
        return cls.instance

    def __init__(self):
        if not hasattr(self, "client"):
            self.client = MonaiLabelClient()
            self._server_monitor = None
            self._server_monitor_thread = None

    def __del__(self):
        self.monitoring_stop()

    def monitoring_stop(self):
        if self._server_monitor is not None:
            self._server_monitor.stop()
            self._server_monitor_thread.quit()
            self._server_monitor_thread.wait()
            self._server_monitor.monitor_stopped.disconnect(
                self._server_monitor_thread.quit
            )
            self._server_monitor_thread.started.disconnect(self._server_monitor.start)

            self._server_monitor.training_started.disconnect(self.training_started)
            self._server_monitor.training_stopped.disconnect(self.training_stopped)
            self._server_monitor.training_progress.disconnect(self.training_progress)

            self._server_monitor_thread = None
            self._server_monitor = None

    def monitoring_start(self):
        if self._server_monitor is not None:
            self.monitoring_stop()
        self._server_monitor = ServerMonitor(deepcopy(self.client))
        self._server_monitor_thread = QThread()
        self._server_monitor.moveToThread(self._server_monitor_thread)
        self._server_monitor.monitor_stopped.connect(self._server_monitor_thread.quit)
        self._server_monitor_thread.started.connect(self._server_monitor.start)

        self._server_monitor.training_started.connect(self.training_started)
        self._server_monitor.training_stopped.connect(self.training_stopped)
        self._server_monitor.training_progress.connect(self.training_progress)

        self._server_monitor_thread.start()

    def set_server(self, url: str):
        self.monitoring_stop()
        success = self.client.set_server(url)
        self.server_changed.emit()
        if success:
            self.monitoring_start()
        else:
            # TODO: show error message
            pass

    def is_connected(self) -> bool:
        return self.client.is_connected()

    @Slot(str)
    def training_start(self, model: str):
        r = self.client.training_start(model)
        print("training_start", r)

    @Slot()
    def training_stop(self):
        r = self.client.training_stop()
        print("training_stop", r)


class ServerMonitor(QObject):
    monitor_stopped = Signal()
    training_started = Signal()
    training_stopped = Signal()
    training_progress = Signal(int, int, datetime.datetime, float)

    def __init__(
        self, client: MonaiLabelClient, *args, poll_interval: float = 2.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.client = client
        self._run = False
        self.poll_interval = poll_interval
        self._is_training = False

    @Slot()
    def start(self):
        self._run = True
        while self._run:
            self._handle_train()
            time.sleep(self.poll_interval)
        self.monitor_stopped.emit()

    @Slot()
    def stop(self):
        self._run = False

    def _handle_train(self):
        train_status = self.client.training_status()
        if self._is_training:
            if train_status is None:
                self._is_training = False
                self.training_stopped.emit()
                # TODO: handle training error
                return
            self._report_train_progress(train_status)
        else:
            if train_status is not None:
                self._is_training = True
                self._max_epochs = None
                self._last_epoch = None
                self._epoch_durations = []
                self.training_started.emit()
                self._report_train_progress(train_status)

    def _report_train_progress(self, train_status):
        details = "\n".join(train_status["details"])
        try:
            if self._max_epochs is None:
                self._max_epochs = int(
                    re.search(r"Epoch: \d*/(\d*), Iter", details).groups()[-1]
                )

            # This text pattern is logged 2x; for train, val phases.
            epochs_durations = re.findall(
                r"Epoch\[(\d*)\] Complete\. Time taken: (.*)",
                details,
            )
            if len(epochs_durations) < 2:
                return

            metric = float(
                re.findall(r"Key metric: val_.* best value: (.*) at epoch", details)[-1]
            )
        except (AttributeError, IndexError):
            return

        epoch, duration = zip(*epochs_durations)

        if not all(i == epoch[0] for i in epoch[1:]):
            return
        epoch = int(epoch[0])
        if epoch == self._last_epoch:
            return

        # Calculate mean epoch duration
        duration = sorted(duration)[-1]
        h, m, s, us = map(
            int, re.match(r"(\d*):(\d{2}):(\d{2}).(\d*)", duration).groups()
        )
        duration = datetime.timedelta(hours=h, minutes=m, seconds=s, microseconds=us)
        duration *= self._max_epochs - epoch
        self._epoch_durations.append(duration)
        duration = np.mean(self._epoch_durations)

        eta = datetime.datetime.now() + duration
        self.training_progress.emit(epoch, self._max_epochs, eta, metric)
        self._last_epoch = epoch
