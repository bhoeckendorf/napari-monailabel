import io
from gzip import GzipFile
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import nrrd
import numpy as np
import requests


class MonaiLabelClient:
    def __init__(self):
        self.url = None
        self._info = None
        self._datastore = None

    def _invalidate(self):
        self._info = None
        self._datastore = None

    def set_server(self, url: str) -> bool:
        self.url = url
        self._invalidate()
        return self.is_connected()

    def is_connected(self) -> bool:
        return self.url is not None and isinstance(self.info, dict)

    @property
    def info(self):
        if self._info is None:
            self._info = self._request("GET", "info").json()
        return self._info

    @property
    def name(self):
        return self.info["name"]

    @property
    def description(self):
        return self.info["description"]

    @property
    def version(self):
        return self.info["version"]

    @property
    def datastore(self):
        if self._datastore is None:
            self._datastore = self._request(
                "GET", "datastore", params={"output": "all"}
            ).json()
        return self._datastore

    def get_datastore_stats(self):
        return self._request("GET", "datastore", params={"output": "stats"}).json()

    @property
    def labels(self) -> List[str]:
        return self.info["labels"]

    @property
    def models(self):
        return self.info["models"]

    def get_model_type(self, model: str) -> str:
        return self.models[model]["type"]

    def get_model_labels(self, model: str) -> Dict[str, int]:
        return self.models[model]["labels"]

    @property
    def trainers(self):
        return self.info["trainers"]

    def get_image_info(self, name: str):
        return self._request(
            "GET", "datastore/image/info", params={"image": name}
        ).json()

    def get_image(self, name: str):
        ext = self.datastore["objects"][name]["image"]["ext"]
        r = self._request("GET", "datastore/image", params={"image": name})
        return _load_image(r.content, ext)

    def get_label_info(self, name: str):
        return self._request(
            "GET", "datastore/label/info", params={"label": name}
        ).json()

    def get_label(self, name: str, tag: str = "final"):
        ext = self.datastore["objects"][name]["labels"][tag]["ext"]
        r = self._request("GET", "datastore/label", params={"label": name, "tag": tag})
        pixels, spacing = _load_image(r.content, ext)
        if pixels.dtype != np.uint8:
            pixels = pixels.astype(np.uint8)
        return pixels, spacing

    def get_available_labels(self, name: str):
        return self.datastore["objects"][name]["labels"]

    def get_active_learning_sample(self, strategy="random"):
        return self._request("POST", f"activelearning/{strategy}").json()

    def infer(self, model: str, image: str):
        ext = self.datastore["objects"][image]["image"]["ext"]
        r = self._request(
            "POST", f"infer/{model}", params={"image": image, "output": "image"}
        )
        pixels, spacing = _load_image(r.content, ext)
        if pixels.dtype != np.uint8:
            pixels = pixels.astype(np.uint8)
        return pixels, spacing

    def training_start(self, model: str):
        return self._request("POST", f"train/{model}").json()

    def training_stop(self):
        return self._request("DELETE", "train").json()

    def is_training(self) -> bool:
        with requests.request(
            "GET", f"{self.url}/train", params={"all": True, "check_if_running": True}
        ) as r:
            if r.status_code == 200:
                v = r.json()
                return v["status"].lower() == "running"
            elif r.status_code == 404:
                return False

    def training_status(self, all: bool = False, check_if_running: bool = True):
        with requests.request(
            "GET",
            f"{self.url}/train",
            params={"all": all, "check_if_running": check_if_running},
        ) as r:
            if r.status_code == 404:
                return None
            return r.json()

    def get_logs(self, num_lines: int = 300, html: bool = False, refresh: int = 0):
        return self._request(
            "GET",
            "logs",
            params={
                "lines": num_lines,
                "html": html,
                "text": not html,
                "refresh": refresh,
            },
        ).text

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None):
        with requests.request(method, f"{self.url}/{endpoint}", params=params) as r:
            if r.status_code == 200:
                return r
            raise ValueError(f"Server response error status code: {r.status_code}")


def _load_image(bytes: bytes, ext: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = ext.strip().lower()
    if ext in (".nii.gz", ".nii"):
        return _load_nifti(bytes, ext)
    elif ext == ".nrrd":
        return _load_nrrd(bytes)
    raise ValueError(f'Unsupported file format: "{ext}"')


def _load_nifti(bytes: bytes, ext: str) -> Tuple[np.ndarray, np.ndarray]:
    buffer = io.BufferedRandom(io.BytesIO(bytes))
    if ext.endswith(".gz"):
        buffer = GzipFile(fileobj=buffer, mode="rb")

    version = None
    x = np.frombuffer(buffer.peek(4), dtype=np.int32, count=1)
    for byteorder in ("<", ">"):
        v = int(x.newbyteorder(byteorder))
        if v == 348:
            version = 1
            break
        elif v == 540:
            version = 2
            break
    if version is None:
        raise ValueError("Could't determine NIFTI header version.")

    if version == 2:
        header = nib.Nifti2Header.from_fileobj(buffer)
        buffer.seek(0)
        fh = nib.Nifti2Image.from_bytes(buffer.read())
    else:
        header = nib.Nifti1Header.from_fileobj(buffer)
        buffer.seek(0)
        fh = nib.Nifti1Image.from_bytes(buffer.read())

    spacing = header["pixdim"][1 : len(header.get_data_shape()) + 1]
    pixels = fh.dataobj.get_unscaled()
    return pixels.transpose(), spacing[::-1]


def _load_nrrd(bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    buffer = io.BytesIO(bytes)

    header = nrrd.read_header(buffer)
    if "spacings" in header:
        spacing = header["spacings"]
    elif "space directions" in header:
        spacing = np.diag(header["space directions"])
    else:
        spacing = np.ones(header["dimension"])
    spacing = spacing[::-1]

    try:
        pixels = nrrd.read_data(header, buffer, index_order="C")
    except io.UnsupportedOperation:
        dtype = header["type"]
        if dtype == "float":
            dtype += "32"
        elif dtype == "double":
            dtype = "float"
        pixels = np.frombuffer(buffer.read(), dtype=np.dtype(dtype)).reshape(
            *header["sizes"][::-1]
        )

    return pixels, spacing
