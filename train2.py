import pyqpanda as pq
from math import *
from pyvqnet.optim import adamax
from pyvqnet import nn
from pyvqnet.tensor import argmax
from pyvqnet.nn.module import Module
from pyvqnet.qnn.quantumlayer import QuantumLayer
import torchvision
from torch.utils.data import DataLoader
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import NLL_Loss
from pyvqnet.tensor import tensor
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
from pyvqnet.tensor import QTensor
from pyvqnet.nn import CrossEntropyLoss
from pyvqnet.nn import SoftmaxCrossEntropy
from pyvqnet.nn import BinaryCrossEntropy
from pyvqnet.nn import MeanSquaredError
from pyvqnet.optim import adadelta
from pyvqnet.optim import SGD
from pyvqnet.optim import adadelta
from pyvqnet.optim import adagrad
from pyvqnet.optim.rmsprop import RMSProp
import numpy as np
import torch
from pyvqnet.nn import Softmax

test_data_size = 50000
# 图像数据增强手段，用于提升识别正确率。

train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),  # 进行概率为0.5的水平翻转，以提高正确率
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

text_data_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),  # 进行概率为0.5的水平翻转，以提高正确率
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

num_train = len(train_data_set)
num_test = len(text_data_set)
train_data_set = torch.utils.data.Subset(train_data_set, indices=range(num_train // 5))
test_data_set = torch.utils.data.Subset(text_data_set, indices=range(num_test // 1))
train_data_load = DataLoader(dataset=train_data_set, batch_size=1, shuffle=True, drop_last=True)
test_data_load = DataLoader(dataset=train_data_set, batch_size=1, shuffle=True, drop_last=True)


def qvc_circuits(input, weights, qlist, clist, machine):
    def get_cnot(nqubits):
        cir = pq.QCircuit()
        for i in range(len(nqubits) - 1):
            cir.insert(pq.CNOT(nqubits[i], nqubits[i + 1]))
        cir.insert(pq.CNOT(nqubits[len(nqubits) - 1], nqubits[0]))
        return cir

    def build_circult(weights, xx, nqubits):

        def Rot(weights_j, qubits):
            circult = pq.QCircuit()
            circult.insert(pq.RZ(qubits, weights_j[0]))
            circult.insert(pq.RY(qubits, weights_j[1]))
            circult.insert(pq.RZ(qubits, weights_j[2]))

            return circult

        def basisstate():
            circult = pq.QCircuit()
            input_state = AmplitudeEmbeddingCircuit(xx, nqubits)
            circult.insert(input_state)

            return circult

        circult = pq.QCircuit()
        circult.insert(basisstate())
        for i in range(weights.shape[0]):

            weights_i = weights[i, :, :]
            for j in range(len(nqubits)):
                weights_j = weights_i[j]
                circult.insert(Rot(weights_j, nqubits[j]))
            cnots = get_cnot(nqubits)
            circult.insert(cnots)
        circult.insert(pq.Z(nqubits[0]))
        prog = pq.QProg()
        prog.insert(circult)
        return prog

    weights = weights.reshape([2, 4, 3])
    prog = build_circult(weights, input, qlist)
    prob = machine.prob_run_dict(prog, qlist, -1)
    prob = list(prob.values())

    return prob


def one_hot_encode(matrix):
    # 创建一个形状为(len(matrix), 10)的零矩阵
    one_hot_encoded = np.zeros((len(matrix), 10))

    # 对于矩阵中的每个元素，将对应的索引位置置为1
    for idx, val in enumerate(matrix):
        one_hot_encoded[idx, val] = 1

    return one_hot_encoded


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2D(3, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLu(),
            nn.MaxPool2D((2, 2), (2, 2), 'valid'),
            nn.BatchNorm2d(32),

            nn.Conv2D(32, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLu(),
            nn.MaxPool2D((2, 2), (2, 2), 'valid'),
            nn.BatchNorm2d(64),

            nn.Conv2D(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLu(),
            nn.MaxPool2D((2, 2), (2, 2), 'valid'),
            nn.BatchNorm2d(128)

        )  # 卷积层(输出通道=3，输出通道=32，填充=1，卷积核=3，通道=1，)-非线性层-池化层

        self.fc = nn.Sequential(

            nn.Linear(2048, 1024),
            nn.ReLu(),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLu(),
            nn.Dropout(),

            nn.Linear(256, 10),

        )
        self.qvc = nn.Sequential(
            QuantumLayer(qvc_circuits, 24, "cpu", 4),

        )

    def forward(self, x):
        x = self.main(x)
        x = tensor.reshape(x, (-1, 2048))
        x = self.fc(x)
        """y=Softmax()
        x=y(x)"""
        return x


model = Model()
optimizer = Adam(model.parameters(), lr=0.001)  # lr=0.001合适
qloss = CrossEntropyLoss()
train_step = 0
test_step = 0
epoch = 20

if __name__ == '__main__':
    for i in range(epoch):  # 一个epoch包括五万次训练和一万次测试
        model.train()
        for j, (imgs, targets) in enumerate(train_data_load):
            optimizer.zero_grad()  # 清零梯度
            outputs = model(imgs)
            outputs=tensor.reshape(outputs,(1,10))
            qtargets = one_hot_encode(targets)
            qtargets = QTensor(qtargets, dtype=5)
            qtargets=tensor.reshape(qtargets,(1,10))

            Qloss = qloss(targets, outputs)

            Qloss.backward()
            optimizer._step()
            train_step += 1  # 每次1张
            if train_step % 100 == 0:
                print(f'训练第{train_step}次,loss={Qloss}')
        accuracy = 0  # 定义准确率
        accuracy_total = 0
        model.eval()
        for j, (imgs, targets) in enumerate(test_data_load):
            outputs = model(imgs)
            qtargets = QTensor(targets)

            if outputs.argmax([1], True) == qtargets:
                accuracy += 1
            targets = one_hot_encode(targets)
            targets = QTensor(targets, dtype=5)
            Qloss_test = qloss(targets, outputs)
            test_step += 1  # 每次1张
            if test_step % 100 == 0:
                print(f'训练第{test_step}次,loss={Qloss_test}')
                print(f'训练第{test_step}次accuracy={accuracy / test_step}')
        del model
