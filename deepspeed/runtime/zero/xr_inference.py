import torch


class XR_param:
    def __init__(self, max):
        self.max = max

    def __iter__(self):
        self.n = 1
        self.param_scope = None
        self.offset = 0
        return self

    def __next__(self):
        if self.n <= self.max:
            self.param_scope = self.set_scope(eval(f"XR_PARAM_SCOPE_SIZE_{self.n}"))
            res = self.param_scope, self.offset, self.n
            self.n += 1
            return res
        else:
            raise StopIteration

    def set_scope(self, size):
        self.param_scope = torch.empty(size,
                                       dtype=torch.float16,
                                       device="cpu").pin_memory()
        return self.param_scope
