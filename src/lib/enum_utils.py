from enum import Enum

class TraningFunctions(Enum):
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'

class TrainingApproaches(Enum):
    SUPERVISED = 'supervised'
    UNSUPERVISED = 'unsupervised'