"""Plugin boundary for CARLA/C2N-Bench or other simulator integrations."""
import importlib
from typing import Protocol


class SimulatorAdapter(Protocol):
    """A benchmark-specific adapter owns sensors, actuators and episode lifecycle."""
    def observe(self) -> dict: ...
    def execute(self, command) -> dict: ...
    def timestamp(self) -> float: ...
    def close(self) -> None: ...


def load_adapter(specification, config):
    module, name = specification.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(config)
