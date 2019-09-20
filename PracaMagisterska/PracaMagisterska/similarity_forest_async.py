import random as rd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import gc
from tree_similarity_forest import Tree


FIT_PROCESSES = 6
PREDICT_PROCESSES = 3

# funkcja tworząca drzewa dla zadanego zbioru danych
def get_tree_async(n_tree, train_dataset, class_dataset, get_subset_func, similarity_func, n_pairs):
    tree = None
    treeSamples, treeClasses = get_subset_func(train_dataset, class_dataset)
    tree = Tree(treeSamples, treeClasses, similarity_func, n_pairs)
    tree.CreateTree()
    return tree.GetCreatedTree()

# funkcja klasyfikująca rekord do odpowiedniej klasy przez głosowanie większościowe
def predict_item_async(item_index, trees, test_dataset, similarity_func, find_root_func, find_node_func):
    predicted = assign_class_to_item(test_dataset[item_index], trees, similarity_func, find_root_func, find_node_func)
    classes = [item["predictClass"] for item in predicted]
    predicted = None
    majorityClass = None
    if len(classes) > 0:
        majorityClassCount = 0
        for label in set(classes):
            if classes.count(label) > majorityClassCount:
                majorityClassCount = classes.count(label)
                majorityClass = label
    return majorityClass

# funkcja sprawdzająca jaką klasę otrzyma rekord w każdym z drzew
def assign_class_to_item(item, trees, similarity_func, find_root_func, find_node_func):
    predicted = []
    for tree in ((index, tree) for index, tree in enumerate(trees)):
        itemClass = None
        node = find_root_func(tree[1])
        while itemClass is None:
            if node is not None:
                if node[0]["leaf"] == True:
                    itemClass = {
                        "predictClass": node[0]["fitClass"],
                        "tree": tree[0]
                    }
                    predicted.append(itemClass)
                    break

                distance = similarity_func(item, node[0]["objectB"]) - similarity_func(item, node[0]["objectA"])
                if distance > node[0]["splitPoint"]:
                    node = find_node_func(tree[1], node[0]["id"], "l_")
                else:
                    node = find_node_func(tree[1], node[0]["id"], "r_")
            else:
                break
    return predicted


class SimilarityForestAsyncClassifier():
    def __init__(self, nTrees, similarityFunction, nPairObjects=1):
        self.nTrees = nTrees
        self.similarityFunction = similarityFunction
        self.nPairObjects = nPairObjects
        self.trees = []
        self.predictions = []
        self.confusionMatrix = []

    # funkcja trenująca zbiór danych, działa na podprocesach
    def fit(self, trainDataset, classDataset):
        print(" ---- START FIT function ---- ")
        if self.nTrees is not None:
            self.trees =  [None] * self.nTrees     
            with mp.Manager() as manager:
                SHARED_TRAIN_DATASET = manager.list(trainDataset)
                SHARED_CLASS_DATASET = manager.list(classDataset)
                with mp.Pool(processes=FIT_PROCESSES) as pool:
                    self.trees = pool.starmap_async(get_tree_async, 
                                                    tqdm([(n, SHARED_TRAIN_DATASET, SHARED_CLASS_DATASET,
                                                           self.GenerateSubset, self.similarityFunction, self.nPairObjects)
                                                          for n in range(self.nTrees)], ncols=50)).get()
                SHARED_TRAIN_DATASET = None
                SHARED_CLASS_DATASET = None
            gc.collect()
            print(" ---- FIT FREE RESOURCES ---- ")
        else:
            print(" Please enter number of trees. ")
            return
        print(" ---- END FIT function ----\n ")

    # funkcja testująca zbiór danych, działa na podprocesach
    def predict (self, testDataset):
        print(" ---- START PREDICT function ---- ")
        predictions = [None] * len(testDataset)
        with mp.Manager() as manager:
            SHARED_TEST_DATASET = manager.list(testDataset)
            SHARED_TREES = manager.list(np.array(self.trees))
            with mp.Pool(processes=PREDICT_PROCESSES) as pool:
                predictions = pool.starmap_async(predict_item_async, tqdm([(n, SHARED_TREES, SHARED_TEST_DATASET, 
                                                                      self.similarityFunction, self.FindRootInTree, self.FindNodeInTree)
                                                                     for n in range(len(testDataset))], ncols=50)).get()
            SHARED_TEST_DATASET = None
            SHARED_TREES = None
        gc.collect()
        print(" ---- PREDICT FREE RESOURCES ---- ")

        print(" ---- END PREDICT function ----\n ")
        return predictions

    # funkcja generująca podzbiór danych dla każdego z drzew
    def GenerateSubset(self, trainDataset, classDataset):
        datasets = []
        if self.nTrees > 1:
            randomSample = rd.choices(range(len(trainDataset)), k=len(trainDataset))
            datasets.append([trainDataset[i] for i in randomSample])
            datasets.append([classDataset[i] for i in randomSample])
        else:
            datasets.append(trainDataset)
            datasets.append(classDataset)
        return datasets

    # funkcja znajdująca korzeń w drzewie
    def FindRootInTree(self, tree):
        root = None
        root = [item for item in tree if item[0]["id"] == "root" and item[0]["parent"] == "root"]
        if len(root) > 0:
            root = root[0]
        return root

    # funkcja znajdująca węzeł w drzewie
    def FindNodeInTree(self, tree, parent, direction):
        node = None
        node = [item for item in tree if item[0]["id"][:2] == direction and item[0]["parent"] == parent]
        if len(node) > 0:
            node = node[0]
        return node