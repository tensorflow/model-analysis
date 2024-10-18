import pytest
from tensorflow.compat.v1 import disable_v2_behavior, enable_v2_behavior

@pytest.fixture(scope="class")
def v2_behavior():
    enable_v2_behavior()
    yield
    disable_v2_behavior()
