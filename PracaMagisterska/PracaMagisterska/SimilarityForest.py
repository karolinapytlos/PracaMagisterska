import random as rd
import numpy as np
import pandas as pd
import math

class SimilarityForest:
    def __init__(self, nTrees, nTreeSamples, similarityFunc):
        self.__nTrees = nTrees
        self.__nTreeSamples = nTreeSamples
        self.__similarityFunction = similarityFunc
        self.__trees = []
        self.__predictions = []
        self.__confusionMatrix = []

    def __generateSubset (self, trainDataset, classDataset):
        datasets = []
        trainSubset = []
        classSubset = []
        if self.__nTrees > 1:
            for x in range(self.__nTreeSamples):
                index = rd.randint(0, (len(trainDataset) -1))
                trainSubset.append(trainDataset[index])
                classSubset.append(classDataset[index])
            datasets.append(trainSubset)
            datasets.append(classSubset)
        else:
            datasets.append(trainDataset)
            datasets.append(classDataset)
        return datasets


    def __findRootInTree (self, tree):
        root = None
        for index, item in enumerate(tree):
            if item["id"] == "root" and item["parent"] == "root":
                root = item
                break
        return root


    def __findNodeInTree (self, tree, parent, direction):
        node = None
        for index, item in enumerate(tree):
            if item["id"].startswith(direction) and item["parent"] == parent:
                node = item
                break
        return node


    def __assignClass (self, item):
        predicted = []
        for index, tree in enumerate(self.__trees):    
            itemClass = None
            node = self.__findRootInTree(tree)
            while itemClass is None:
                if node is not None:
                    if node["leaf"] == True:
                        itemClass = {
                            "predictClass": node["fitClass"],
                            "tree": index
                            }
                        predicted.append(itemClass)
                        break

                    distance = self.__similarityFunction(item, node["objectB"]) - self.__similarityFunction(item, node["objectA"])
                    if distance > node["splitPoint"]:
                        node = self.__findNodeInTree(tree, node["id"], "l_")
                    else:
                        node = self.__findNodeInTree(tree, node["id"], "r_")
                else:
                    break
        return predicted


    def fit (self, trainDataset, classDataset):
        for n in range(self.__nTrees):
            print(" ---- START TREE: ", n, " ---- ")
            treeSamples, treeClasses = self.__generateSubset(trainDataset, classDataset)
            tree = Tree(treeSamples, treeClasses, self.__similarityFunction)
            tree.create_tree()
            self.__trees.append(tree.get_tree())
            print(" ---- END TREE: ", n, " ---- ")


    def get_confusion_matrix (self, classDataset):
        print(" ---- START ---- ")
        classes = list(set(classDataset))
        self.__confusionMatrix = np.array([[0, 0], [0, 0]])
        for index, value in enumerate(classDataset):
            predicted = [item for item in self.__predictions if item["index"] == index]
            if len(predicted) > 0:
                predicted = predicted[0]
                if  classes[0] == value:
                    if value == predicted["class"]: # TP
                        self.__confusionMatrix[0, 0] = self.__confusionMatrix[0, 0] +1
                    else: # FN
                        self.__confusionMatrix[0, 1] = self.__confusionMatrix[0, 1] +1
                if classes[1] == value:
                    if value == predicted["class"]: # TN
                        self.__confusionMatrix[1, 1] = self.__confusionMatrix[1, 1] +1
                    else: # FP
                        self.__confusionMatrix[1, 0] = self.__confusionMatrix[1, 0] +1
        print(" ---- END ---- ")
        return pd.DataFrame.from_dict({ classes[0]: self.__confusionMatrix[0], classes[1]: self.__confusionMatrix[1] }, orient='index', columns=classes)


    def predict (self, testDataset):
        print(" ---- START ---- ")
        for index, value in enumerate(testDataset):
            predicted = self.__assignClass(value)
            classes = [item["predictClass"] for item in predicted]
            if len(classes) > 0:
                majorityClass = None
                majorityClassCount = 0
                for i, label in enumerate(set(classes)):
                    if classes.count(label) > majorityClassCount:
                        majorityClassCount = classes.count(label)
                        majorityClass = label
                if majorityClass is not None:
                    self.__predictions.append({ "index": index, "class": majorityClass })
        print(" ---- END ---- ")


    def get_predictions (self):
        return self.__predictions



class Tree:
    def __init__(self, dataset, labels, similarityFunc):
        self.__dataset = dataset
        self.__labels = labels
        self.__similarityFunction = similarityFunc
        self.__tree = []
        self.__countLeftNode = 0
        self.__countRightNode = 0


    def __randLabelIndex (self, labelsDataset):
        indexes = []
        for item in set(labelsDataset):
            labelList = self.__filterList(labelsDataset, item)
            if len(labelList) > 0:
                indexes.append(rd.choice(labelList))
        return indexes


    def __filterList (self, dataset, item):
        filtered = []
        for index, value in enumerate(dataset):
            if  value == item:
                filtered.append((index, value))
        return filtered


    def __countValues (self, dataset, item):
        counter = 0
        for index, value in enumerate(dataset):
            if  value == item:
                counter = counter +1
        return counter


    def __create (self, node):
        distances = []
        dataset = node["data"]
        labels = node["labels"]

        if len(set(labels)) == 1:
            self.__setNodeAsLeaf(node, labels[0])
            return

        # select randomly item from every class
        # item is type of tuple
        # item 1 is index, item 2 is class value
        labelesIndex = self.__randLabelIndex(labels)

        randItems = []
        for t in labelesIndex:
            randItems.append(dataset[t[0]])

        # calculate distance between items from dataset
        for index, value in enumerate(dataset):
            dist = self.__similarityFunction(value, randItems[1]) - self.__similarityFunction(value, randItems[0])
            distances.append((index, dist))

        distances = sorted(distances, key=lambda tup: tup[1])

        ds = []
        lb = []
        for item in distances:
            ds.append(dataset[item[0]])
            lb.append(labels[item[0]])

        gini = []
        for index in range(len(lb) -1):
            node1 = lb[:index+1]
            node2 = lb[index+1:]
            quality = self.__giniQuality(node1, node2)
            point = {
                "index": index,
                "gq": quality
            }
            gini.append(point)

        splitPoint = min(gini, key = lambda x: x['gq'])

        node["objectA"] = randItems[0]
        node["objectB"] = randItems[1]
        node["splitPoint"] = splitPoint["gq"]

        leftNodeData = []
        leftNodeLabels = []
        rightNodeData = []
        rightNodeLabels = []
        for index, item in enumerate(distances):
            if item[1] > splitPoint['gq']:
                leftNodeData.append(ds[index])
                leftNodeLabels.append(lb[index])
            else:
                rightNodeData.append(ds[index])
                rightNodeLabels.append(lb[index])


        if len(leftNodeData) > 1 and len(rightNodeData) < 1:
            rightNodeData.append(leftNodeData[len(leftNodeData) -1])
            rightNodeLabels.append(leftNodeLabels[len(leftNodeLabels) -1])
            leftNodeData = leftNodeData[:-1]
            leftNodeLabels = leftNodeLabels[:-1]

        if len(leftNodeData) < 1 and len(rightNodeData) > 1:
            leftNodeData.append(rightNodeData[len(rightNodeData) -1])
            leftNodeLabels.append(rightNodeLabels[len(rightNodeLabels) -1])
            rightNodeData = rightNodeData[:-1]
            rightNodeLabels = rightNodeLabels[:-1]


        nodeLeft = None
        if len(leftNodeData) > 0:
            self.__countLeftNode += 1
            nodeLeft = {
                "id": "l_" + str(self.__countLeftNode),
                "parent": node["id"],
                "data": leftNodeData,
                "labels": leftNodeLabels,
                "leaf": False,
                "objectA": None,
                "objectB": None,
                "splitPoint": None,
                "fitClass": None
            }
            self.__tree.append(nodeLeft)

        nodeRight = None
        if len(rightNodeData) > 0:
            self.__countRightNode += 1
            nodeRight = {
                "id": "r_" + str(self.__countRightNode),
                "parent": node["id"],
                "data": rightNodeData,
                "labels": rightNodeLabels,
                "leaf": False,
                "objectA": None,
                "objectB": None,
                "splitPoint": None,
                "fitClass": None
            }
            self.__tree.append(nodeRight)
        

        if nodeLeft is not None:
            self.__create(nodeLeft)

        if nodeRight is not None:
            self.__create(nodeRight)

        if nodeLeft is None and nodeRight is None:
            return


    def __giniQuality (self, node1, node2):
        # quantity of items in node1
        rNode1 = len(node1)
        # quantity of items in node2
        rNode2 = len(node2)

        # value of gini index for node1
        pNode1 = self.__gini(node1)
        # value of gini index for node2
        pNode2 = self.__gini(node2)

        # value of weighted gini quality GQ(N1, N2)
        quality = (pNode1 + pNode2) / (rNode1 + rNode2)

        return quality


    def __gini (self, node):
        items = len(node)

        occurrences = []
        for label in set(node):
            attribute = {
                "class": label,
                "count": self.__countValues(node, label)
            }
            occurrences.append(attribute)

        # calculate value of gini index
        probability = 1
        for item in occurrences:
            value = math.pow((item["count"] / items), 2)
            probability = probability - value

        return probability


    def __setNodeAsLeaf (self, node, label):
        node["leaf"] = True
        node["fitClass"] = label


    def create_tree (self):
        root = {
            "id": "root",
            "parent": "root",
            "data": self.__dataset,
            "labels": self.__labels,
            "leaf": False,
            "objectA": None,
            "objectB": None,
            "splitPoint": None,
            "fitClass": None
        }
        self.__tree.append(root)
        self.__create(root)

        return


    def get_tree (self):
        return self.__tree


    def display_tree (self):
        print(self.__tree)