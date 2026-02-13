from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

class BaseLLMModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: Any, **kw_args: Any) -> Any: ...

    @abstractmethod
    def load_weights(self, *args: Any, **kw_args: Any) -> Any: ...
