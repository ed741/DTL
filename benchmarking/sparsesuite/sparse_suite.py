from dataclasses import dataclass

import numpy as np
import scipy
from numpy import ndarray

from benchmarking.benchmark import Benchmark


@dataclass
class TensorWrapper:
    tensor: ndarray | scipy.sparse.coo_array

    def get_as_array(self) -> ndarray | tuple[tuple[ndarray], ndarray]:
        if isinstance(self.tensor, ndarray):
            return self.tensor
        elif isinstance(self.tensor, scipy.sparse.coo_array):
            return self.tensor.coords, self.tensor.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.tensor, scipy.sparse.coo_array)

    @property
    def non_zeros(self) -> int:
        if isinstance(self.tensor, scipy.sparse.coo_array):
            return self.tensor.nnz
        else:
            return np.prod(self.shape)


def handle_tensor(
    benchmark: Benchmark, data_set_name: str, name: str, is_arg: bool, is_res: bool, ref_tensor
) -> TensorWrapper:
    if scipy.sparse.issparse(ref_tensor):
        ref_tensor = scipy.sparse.coo_array(ref_tensor).astype(np.float32)
        nnz = ref_tensor.nnz
        for i, coord_array in enumerate(ref_tensor.coords):
            benchmark.handle_reference_array(
                coord_array.astype(np.int32),
                f"arrays/{data_set_name}/np_{name}_{i}",
                is_arg,
                is_res,
                f"{name}_{i}",
                binary=True,
                dtype=np.int32,
            )
            assert coord_array.shape == (nnz,)
        benchmark.handle_reference_array(
            ref_tensor.data,
            f"arrays/{data_set_name}/np_{name}_val",
            is_arg,
            is_res,
            f"{name}_val",
            binary=True,
            dtype=np.float32
        )
        assert ref_tensor.data.shape == (nnz,)
        return TensorWrapper(ref_tensor)
    else:
        ref_tensor = np.array(ref_tensor).astype(np.float32)
        benchmark.handle_reference_array(
            ref_tensor,
            f"arrays/{data_set_name}/np_{name}",
            is_arg,
            is_res,
            name,
            binary=True,
        )
        return TensorWrapper(ref_tensor)

