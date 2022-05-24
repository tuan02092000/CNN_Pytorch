import config
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import argparse
import imutils
import torch
import cv2

np.random.seed(42)

# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to the trained model")
args = vars(ap.parse_args())

# load the MNIST dataset and randomly grab 10 data points
print("[INFO] loading to test dataset...")
testData = MNIST(root="data",
                 train=False,
                 download=True,
                 transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10, ))
testData = Subset(testData, idxs)

# initialize test dataloader
testDataLoader = DataLoader(testData, batch_size=1)

# load model and set it to evaluation mode
model = torch.load(args["model"]).to(config.DEVICE)
model.eval()

# switch off autograd
with torch.no_grad():
    # loop over the test set
    for (image, label) in testDataLoader:
        origImage = image.numpy().squeeze(axis=(0, 1))
        gtLabel = testData.dataset.classes[label.numpy()[0]]

        image = image.to(config.DEVICE)
        pred = model(image)

        idx = pred.argmax(1).cpu().numpy()[0]
        predLabel = testData.dataset.classes[idx]

        origImage = np.dstack([origImage] * 3)
        origImage = imutils.resize(origImage, width=128)

        color = (255, 0, 0) if gtLabel == predLabel else (0, 255, 0)
        cv2.putText(origImage, gtLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

        print("[INFO] ground truth label: {}, predicted label: {}".format(
            gtLabel, predLabel))
        cv2.imshow("image", origImage)
        cv2.waitKey(0)

