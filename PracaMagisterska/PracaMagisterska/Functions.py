import math
import numpy as np


def cosie_similarity (vectorA, vectorB):
    return np.dot(vectorA, vectorB) / (math.sqrt(np.dot(vectorA, vectorA)) * math.sqrt(np.dot(vectorB, vectorB)))