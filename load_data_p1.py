import os
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

print("load_data_p1.py: Started")

is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if "asl_alphabet_train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/asl_alphabet_finalproject/asl_alphabet.zip")
    os.system("unzip asl_alphabet.zip")

#os.getcwd()  ---->  /home/ubuntu/final_project_ml2
DATA_DIR = os.getcwd() + "/asl_alphabet_train/"

aslDataset = torchvision.datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

batch_size = 5800

aslLoader = torch.utils.data.DataLoader(aslDataset, batch_size=batch_size, shuffle=True)
dataIter=iter(aslLoader)
inputs, classes = dataIter.next()

class_labels = aslLoader.dataset.classes

np.save('ASLimages', inputs)
np.save('ASLlabels', classes)

print("Data saved into two numpy file files:")
print("ASLimages.npy")
print("ASLlabels.npy")

print("load_data_p1.py: Completed")