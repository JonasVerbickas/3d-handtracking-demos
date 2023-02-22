
from enum import Enum

class KeypointEstimatorEnum(Enum):
    NONE = 0
    MEDIAPIPE = 1
    MESHFORMER = 2

MODELS_THAT_REQUIRE_PALM_DETECTION = [KeypointEstimatorEnum.MESHFORMER]