# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function

from collections import namedtuple
from ctypes import cdll, c_char_p
from typing import Any, Sequence, Tuple, Type

import numpy as np
import tensorrt as trt

from .tensorrt_engine import Engine

libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def namedtupledict(
        typename: str, field_names: Sequence[str], *args: Any, **kwargs: Any
) -> Type[Tuple[Any, ...]]:
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault("rename", True)
    data = namedtuple(typename, field_names, *args, **kwargs)  # type: ignore

    def getitem(self: Any, key: Any) -> Any:
        if isinstance(key, str):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)  # type: ignore

    data.__getitem__ = getitem  # type: ignore[assignment]
    return data


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        if isinstance(error_string, bytes):
            error_string = error_string.decode("utf-8")
        raise RuntimeError("cudaSetDevice: " + error_string)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTBackendRep():
    def __init__(self, model, device_id):
        cudaSetDevice(device_id)
        self._logger = TRT_LOGGER
        self.model = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)
        with self._logger as logger, trt.Runtime(logger) as runtime:
            with open(self.model, mode='rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.engine = Engine(engine)

    def run(self, inputs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        outputs = self.engine.run(inputs)
        output_names = [output.name for output in self.engine.outputs]
        return namedtupledict('Outputs', output_names)(*outputs)
