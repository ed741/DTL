from typing import Dict

import numpy as np

import dtl.dag


class iastNode(dtl.astNode):
    def evaluate(tensors: Dict[str, type(np.narray)]) -> type(np.narray):
        raise TypeError
