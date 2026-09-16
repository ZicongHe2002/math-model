"""mla_model — parameter-driven mathematical model of MLA forward.

Core is CPU-only: numpy + sympy.  No JAX / TPU / LLM dependency.
"""
from .model import MLAForwardModel, BoundModel, comparison_to_markdown
from .report import ABSENT, ABSENT_JSON, Report, Metric, is_absent
from .metadata import TensorMeta, parse_tensor_metadata
from .semantics import SemanticConfig
from .numerics import NumericPolicy
from .implementation import ExecutionStrategy
from .hardware import HardwareProfile
from .errors import (MLAModelError, BindingError, CapabilityError, MetadataError, UnsupportedDtypeError,
                     TaskIdentityError, ReferenceSemanticsError)

__version__ = "0.1.0"
__all__ = ["MLAForwardModel", "BoundModel", "comparison_to_markdown", "Report", "Metric",
           "ABSENT", "ABSENT_JSON", "is_absent", "TensorMeta", "parse_tensor_metadata",
           "SemanticConfig", "NumericPolicy", "ExecutionStrategy", "HardwareProfile", "MLAModelError", "BindingError",
           "CapabilityError", "MetadataError", "UnsupportedDtypeError", "TaskIdentityError", "ReferenceSemanticsError"]
