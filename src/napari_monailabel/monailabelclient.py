import io
from gzip import GzipFile

import nibabel as nib
import nrrd
import numpy as np
import requests


class MonaiLabelClient:

    def __init__(self):
        self._url = None
        self._info = None
        self._datastore = None

    def _invalidate(self):
        self._info = None
        self._datastore = None

    def connect(self, url):
        self._url = url
        self._invalidate()

    @property
    def info(self):
        if self._info is None:
            with requests.get(f"{self._url}/info") as r:
                self._info = r.json()
        return self._info

    @property
    def datastore(self):
        if self._datastore is None:
            with requests.get(f"{self._url}/datastore", params={"output": "all"}) as r:
                self._datastore = r.json()
        return self._datastore

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
    def labels(self):
        return self.info["labels"]

    @property
    def models(self):
        return self.info["models"]

    def get_image_info(self, name: str):
        with requests.get(f"{self._url}/datastore/image/info", params={"image": name}) as r:
            return r.json()

    def get_image(self, name: str):
        ext = self.datastore["objects"][name]["image"]["ext"]
        with requests.get(f"{self._url}/datastore/image", params={"image": name}) as r:
            return _load_image(r.content, ext)

    def get_label_info(self, name: str):
        with requests.get(f"{self._url}/datastore/label/info", params={"label": name}) as r:
            return r.json()

    def get_label(self, name: str):
        ext = self.datastore["objects"][name]["label"]["final"]["ext"]
        with requests.get(f"{self._url}/datastore/label", params={"label": name}) as r:
            return _load_image(r.content, ext)

    def get_active_learning_sample(self, strategy="random"):
        with requests.post(f"{self._url}/activelearning/{strategy}") as r:
            return r.json()

    def infer(self, model: str, image: str):
        ext = self.datastore["objects"][image]["image"]["ext"]
        assert ext == ".nrrd"
        with requests.post(f"{self._url}/infer/{model}", params={"image": image, "output": "image"}) as r:
            bytes = io.BufferedRandom(io.BytesIO(r.content))
            header = nrrd.read_header(bytes)
            msk = np.frombuffer(bytes.read(), dtype=np.float32)
        msk = msk.reshape(*header["sizes"][::-1]).astype(np.uint8)
        return msk, _get_nrrd_spacing(header)


def _nifti_version(bytes) -> int:
    x = np.frombuffer(bytes.peek(4), dtype=np.int32, count=1)
    for byteorder in ("<", ">"):
        i = int(x.newbyteorder(byteorder))
        if i == 348:
            return 1
        elif i == 540:
            return 2
        raise ValueError()


def _load_nifti(bytes):
    stream = GzipFile(fileobj=io.BufferedRandom(io.BytesIO(bytes)), mode="rb")
    if _nifti_version(stream) == 2:
        stream.seek(0)
        header = nib.Nifti2Header.from_fileobj(stream)
        stream.seek(0)
        fh = nib.Nifti2Image.from_bytes(stream.read())
    else:
        stream.seek(0)
        header = nib.Nifti1Header.from_fileobj(stream)
        stream.seek(0)
        fh = nib.Nifti1Image.from_bytes(stream.read())
    spc = header["pixdim"][1:len(header.get_data_shape()) + 1][::-1]
    pix = fh.dataobj.get_unscaled().transpose()
    return pix, spc


def _get_nrrd_spacing(header):
    if "space directions" in header:
        out = np.diag(header["space directions"])
    else:
        out = header["spacings"]
    return out[::-1]


def _load_nrrd(bytes):
    buffer = io.BufferedRandom(io.BytesIO(bytes))
    header = nrrd.read_header(buffer)
    return nrrd.read_data(header, buffer, index_order="C"), _get_nrrd_spacing(header)


def _load_image(bytes, ext):
    ext = ext.strip().lower()
    if ext == ".nii.gz":
        return _load_nifti(bytes)
    elif ext == ".nrrd":
        return _load_nrrd(bytes)
    raise ValueError()
