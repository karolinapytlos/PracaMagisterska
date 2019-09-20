import random as rd
import numpy as np
import math
import sys
import os


NODE_TYPE = np.dtype([("id", np.object), ("parent", np.object),
                      ("data", np.object), ("labels", np.object),
                      ("leaf", np.bool), ("objectA", np.object),
                      ("objectB", np.object), ("splitPoint", np.float),
                      ("fitClass", np.object)])

SPLIT_POINT_TYPE = np.dtype([("objectA", np.object), ("objectB", np.object),
                             ("splitPoint", np.float), ("distances", np.object),
                             ("ds", np.object), ("lb", np.object)])

# klasa tworząca drzewo
# polega na rekurencyjnym dzieleniu zbioru danych i tworzeniu kolejnych węzłów,
# tak aby uzyskać węzły, które zawierają konkretne klasy
class Tree:
    def __init__(self, dataset, labels, similarityFunc, nPairObjects):
        if type(dataset) is not np.ndarray:
            self.dataset = np.array(dataset)
        else:
            self.dataset = dataset
        if type(labels) is not np.ndarray:
            self.labels = np.array(labels)
        else:
            self.labels = labels
        self.similarityFunction = similarityFunc
        self.nPairObjects = nPairObjects
        self.tree = []
        self.countLeftNode = 0
        self.countRightNode = 0


    def CreateTree (self):
        root = np.empty(1, dtype=NODE_TYPE)
        root[0]["id"] = "root"
        root[0]["parent"] = "root"
        root[0]["data"] = self.dataset
        root[0]["labels"] = self.labels
        root[0]["leaf"] = False
        root[0]["objectA"] = None
        root[0]["objectB"] = None
        root[0]["splitPoint"] = None
        root[0]["fitClass"] = None

        self.tree.append(root)
        self.__Create(root[0])
        return


    def GetCreatedTree (self):
        return np.array(self.tree)


    def __Create (self, node):
        if len(set(node["labels"])) == 1:
            self.__SetNodeAsLeaf(node, node["labels"][0])
            return

        nodeSplitPoint = self.__GetNodeSplitPoint(node["data"], node["labels"])
        if nodeSplitPoint is None:
            return

        node["objectA"] = nodeSplitPoint["objectA"][0]
        node["objectB"] = nodeSplitPoint["objectB"][0]
        node["splitPoint"] = nodeSplitPoint["splitPoint"]

        leftNodeData = []
        leftNodeLabels = []
        rightNodeData = []
        rightNodeLabels = []
        for item in ((int(item[0]), item[1]) for item in nodeSplitPoint["distances"][0]):
            if item[1] > nodeSplitPoint["splitPoint"]:
                leftNodeData.append(nodeSplitPoint["ds"][0][item[0]])
                leftNodeLabels.append(nodeSplitPoint["lb"][0][item[0]])
            else:
                rightNodeData.append(nodeSplitPoint["ds"][0][item[0]])
                rightNodeLabels.append(nodeSplitPoint["lb"][0][item[0]])

        if len(leftNodeData) > 1 and len(rightNodeData) < 1:
            rightNodeData.append(leftNodeData[len(leftNodeData) - 1])
            rightNodeLabels.append(leftNodeLabels[len(leftNodeLabels) - 1])
            leftNodeData = leftNodeData[:-1]
            leftNodeLabels = leftNodeLabels[:-1]

        if len(leftNodeData) < 1 and len(rightNodeData) > 1:
            leftNodeData.append(rightNodeData[len(rightNodeData) - 1])
            leftNodeLabels.append(rightNodeLabels[len(rightNodeLabels) - 1])
            rightNodeData = rightNodeData[:-1]
            rightNodeLabels = rightNodeLabels[:-1]

        nodeLeft = None
        if len(leftNodeData) > 0:
            self.countLeftNode += 1
            nodeLeft = np.empty(1, dtype=NODE_TYPE)
            nodeLeft[0]["id"] = "{0}_{1}".format("l", self.countLeftNode)
            nodeLeft[0]["parent"] = node["id"]
            nodeLeft[0]["data"] = leftNodeData
            nodeLeft[0]["labels"] = leftNodeLabels
            nodeLeft[0]["leaf"] = False
            nodeLeft[0]["objectA"] = None
            nodeLeft[0]["objectB"] = None
            nodeLeft[0]["splitPoint"] = None
            nodeLeft[0]["fitClass"] = None

            self.tree.append(nodeLeft)
            nodeLeft = nodeLeft[0]["id"]
            leftNodeData = None
            leftNodeLabels = None

        nodeRight = None
        if len(rightNodeData) > 0:
            self.countRightNode += 1
            nodeRight = np.empty(1, dtype=NODE_TYPE)
            nodeRight[0]["id"] = "{0}_{1}".format("r", self.countRightNode)
            nodeRight[0]["parent"] = node["id"]
            nodeRight[0]["data"] = rightNodeData
            nodeRight[0]["labels"] = rightNodeLabels
            nodeRight[0]["leaf"] = False
            nodeRight[0]["objectA"] = None
            nodeRight[0]["objectB"] = None
            nodeRight[0]["splitPoint"] = None
            nodeRight[0]["fitClass"] = None

            self.tree.append(nodeRight)
            nodeRight = nodeRight[0]["id"]
            rightNodeData = None
            rightNodeLabels = None

        if nodeLeft is not None:
            self.__Create(next((node for node in self.tree if node[0]["id"] == nodeLeft))[0])

        if nodeRight is not None:
            self.__Create(next((node for node in self.tree if node[0]["id"] == nodeRight))[0])

        if nodeLeft is None and nodeRight is None:
            return


    def __SetNodeAsLeaf (self, node, label):
        node["leaf"] = True
        node["fitClass"] = label

    # funkcja generująca punkt podziału zbioru
    def __GetNodeSplitPoint (self, dataset, labels):
        splitPoints = []
        for i in range(self.nPairObjects):
            newSplitPoint = np.empty(1, dtype=SPLIT_POINT_TYPE)

            # losowo wybierz dwa punkty ze zbioru danych
            labelsIndexes = self.__ChooseRandomlyTwoIndexesWithDifferentLabels(labels)

            randomItems = self.__GetItemsFromDatasetByIndexes(dataset, [item[0] for item in labelsIndexes])

            # wylicz dystans pomiędzy resztą ze zbioru danych, a dwoma wcześniej wybranymi punktami
            distances = self.__CalculateDistancesBetweenChosenItemsAndItemsFromDataset(dataset, randomItems)

            ds = []
            lb = []
            for item in distances:
                ds.append(dataset[item[0]])
                lb.append(labels[item[0]])

            # dla każdego możliwego przedziału wylicz ważony Index Gini
            gini = []
            for index in range(len(lb) - 1):
                gini.append(self.__CalculateWeightedGiniQuality(lb[:index + 1], lb[index + 1:]))

            newSplitPoint[0]["objectA"] = randomItems[0]
            newSplitPoint[0]["objectB"] = randomItems[1]
            newSplitPoint[0]["splitPoint"] = min(gini)
            newSplitPoint[0]["distances"] = np.array(distances)
            newSplitPoint[0]["ds"] = np.array(ds)
            newSplitPoint[0]["lb"] = np.array(lb)
            splitPoints.append(newSplitPoint)

        newSplitPoint = None
        labelsIndexes = None
        randomItems = None
        distances = None
        ds = None
        lb = None
        gini = None

        if len(splitPoints) > 0:
            # punkt podziału będzie tam, gdzie najmniejsza jego wartość
            return min(splitPoints, key=lambda x: x["splitPoint"])
        else:
            return None


    def __ChooseRandomlyTwoIndexesWithDifferentLabels (self, labels):
        indexes = []
        for label in set(labels):
            indexes.append(rd.choice([item for item in ((index, item) for (index, item) in enumerate(labels) if item == label)]))
        return np.array(indexes)

    
    def __GetItemsFromDatasetByIndexes (self, dataset, indexes):
        return np.array([item for item in (item for (index, item) in enumerate(dataset) if index in indexes)])


    def __CalculateDistancesBetweenChosenItemsAndItemsFromDataset (self, dataset, randomItems):
        distances = [(item[0], self.similarityFunction(item[1], randomItems[1]) - self.similarityFunction(item[1], randomItems[0])) 
                     for item in ((index,value) for index, value in enumerate(dataset))]
        return sorted(distances, key=lambda tup: tup[1])


    def __CalculateWeightedGiniQuality (self, node1, node2):
        return ((self.__CalculateGiniIndex(node1) + self.__CalculateGiniIndex(node2)) / (len(node1) + len(node2)))


    def __CalculateGiniIndex (self, node):
        occurrences = []
        for label in set(node):
            occurrences.append((label, len([i for i in (item for item in node if label == item)])))
        # calculate gini index
        probability = 1
        for item in occurrences:
            probability = probability - math.pow((item[1] / len(node)), 2)
        return probability