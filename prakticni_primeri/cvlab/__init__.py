"""cvlab — interactive explorations of computer-vision exam questions.

The library knows nothing about hosts: an example is a pure function
(params) -> ordered dict of outputs, registered with metadata.
"""

from cvlab.core import REGISTRY, Bool, Choice, Example, Float, Int, example, load_examples

__all__ = ["REGISTRY", "Bool", "Choice", "Example", "Float", "Int", "example", "load_examples"]
