import math
import numpy as np
from scipy.spatial import distance
import stringdist as sd


def cosine_distance (vectorA, vectorB):
    if (math.sqrt(np.dot(vectorA, vectorA)) * math.sqrt(np.dot(vectorB, vectorB))) != 0:
        return np.dot(vectorA, vectorB) / (math.sqrt(np.dot(vectorA, vectorA)) * math.sqrt(np.dot(vectorB, vectorB)))
    else:
        return 0


# SOURCE: https://www.geeksforgeeks.org/longest-common-subsequence-dp-4
def longest_common_subsequence(sequenceA, sequenceB):
    # find the length of the strings 
    m = len(sequenceA) 
    n = len(sequenceB)
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in range(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif sequenceA[i-1] == sequenceB[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n]


def euclidean_distance (vectorA, vectorB):
    return distance.euclidean(vectorA, vectorB)


def squared_euclidean_distance (vectorA, vectorB):
    return distance.sqeuclidean(vectorA, vectorB)


def levenshtein_distance (vectorA, vectorB):
    return sd.levenshtein(vectorA, vectorB)