import pandas as pd
import numpy as np
from numpy import genfromtxt

class class_dataset:
    X = []
    y = []
    name = ""

    def __init__(self, dataset_name=None):
        self.name = dataset_name
        self.pickDataset()

    def pickDataset(self):
        if self.name == "diabetes":
            self.diabetesDataset()
        elif self.name == "2D":
            self.artificialDataset()
        elif self.name == "glcm":
            self.odilDataset()
        elif self.name == "dissimilarity":
            self.tenun_dissimilarity()
        elif self.name == "alkohol":
            self.alkoholDataset()

    def alkoholDataset(self):
        print("DATASET ALKOHOL")
        df = genfromtxt('wine-clustering.csv', delimiter=',', skip_header=1)
        self.X = df[:, 1:13]
        y = df[:, 0:1]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy

    def artificialDataset(self):
        print("DATASET 2D")
        file = pd.read_excel(open('dataset_artificial.xlsx', 'rb'))
        X = pd.DataFrame(file, columns=(['x', 'y']))
        self.X = np.array(X)
        y = pd.DataFrame(file, columns=(['T']))
        y = np.array(y)
        yy = []
        for i in range(len(y)):
            yy.append(y[i][0])
        self.y = yy

    def diabetesDataset(self):
        print("DATASET DIABETES")
        df = genfromtxt('diabetes_dataset.csv.xls', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 0:8]
        y = df[:, 8:9]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy

    def odilDataset(self):
        print("DATASET glcm")
        df = genfromtxt('GLCM.csv.xls', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 1:25]
        y = df[:, 25:26]
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy

    def tenun_dissimilarity(self):
        print("DATASET Dissimilarity")
        df = genfromtxt('GLCMbaru.csv.xls', delimiter=',', skip_header=1)
        # print(df.shape)
        self.X = df[:, 1:5]
        y = df[:, 25:26]
        print(y)
        yy = []
        for i in range(len(y)):
            # print(y[i][0])
            yy.append(y[i][0])
        self.y = yy