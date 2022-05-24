import config
from utils.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch import nn
import numpy as np
import argparse
import torch
import time
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# load the dataset
print("[INFO] loading the MNIST dataset...")
trainData = MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
testData = MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * config.TRAIN_SPLIT)
numValSamples = int(len(trainData) * config.VAL_SPLIT)
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples], generator=torch.Generator().manual_seed(42))

# initialize the train, validation and test data_loader
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=config.BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=config.BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=config.BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // config.BATCH_SIZE
valSteps = len(valDataLoader.dataset) // config.BATCH_SIZE

# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(config.DEVICE)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=config.INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, config.EPOCHS):
    model.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for (x, y) in trainDataLoader:
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for (x, y) in valDataLoader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # calculate the average training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(valCorrect)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e, config.EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.6f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.6f}".format(avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

print("[INFO] evaluating network")
# turn off autograd for testing evaluation
with torch.no_grad():
    model.eval()

    # initialize a list to store our predictions
    preds = []

    # loop over the test set
    for (x, y) in testDataLoader:
        x = x.to(config.DEVICE)
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
# generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
                            np.array(preds),
                            target_names=testData.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])
