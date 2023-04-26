import torch
import hashlib
from typing import Iterable
from torch.nn import Parameter


class xr_param:
    def __init__(self, key_dict):
        self.key_dict = {}

    def __getattr__(self, key):
        return self.key_dict[key]

    def _set_key(self, param: Iterable[Parameter]):
        tk = hash(tuple(p.ds_id for p in param)) % 256
        return tk

    def set_map(self, param: Iterable[Parameter]):
        # consistent hash
        tk = self._set_key(param)
        if len(param) == 1:
            self.key_dict[tk] = param.ds_tensor.pin_memory()
        else:
            for p in param:
                self.key_dict[tk] = torch.cat(
                    [p.ds_tensor for p in params]
                ).pin_memory()

    def param_copy(partitions: tensor, param: Iterable[Parameter]):
        if tuple(p.ds_id for p in param) not in self.key_dict:
            self.__setattr__(param)
        partitions.copy_(self.key_dict[tuple(p.ds_id for p in param)])
        return partitions

    def free_param():
        pass
