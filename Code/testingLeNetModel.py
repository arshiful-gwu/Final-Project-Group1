import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from time import clock
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pandas as pd

print("testingLeNetModel.py: started")
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

writer_train = SummaryWriter('finalrunsLNTesting/models')

batch_size = 64
epochs = 4

is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

class aslCustomDataset(Dataset):
    def __init__(self, filepathImages, filepathLabels, transform=None):
        self.imagespath = filepathImages
        self.labelspath = filepathLabels

        self.dataImages = np.load(self.imagespath, mmap_mode='r')
        self.dataLabels = np.load(self.labelspath, mmap_mode='r')
        self.transform = transform

    def __len__(self):

        return len(self.dataLabels)

    def __getitem__(self, index):

        image = self.dataImages[index].transpose()
        label = self.dataLabels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

imagetrainpath = 'file_train_Images.npy'
labeltrainpath = 'file_train_Labels.npy'

train_dataset = aslCustomDataset(filepathImages=imagetrainpath,
                                 filepathLabels=labeltrainpath,
                                 transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_iter = iter(train_loader)
print(type(train_iter))

images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

out = torchvision.utils.make_grid(images)

imshow(out, title=[labels])

imagevalpath = 'file_test_Images.npy'
labelvalpath = 'file_test_Labels.npy'

val_dataset = aslCustomDataset(filepathImages=imagevalpath,
                                 filepathLabels=labelvalpath,
                                 transform=transforms.ToTensor())

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

val_iter = iter(val_loader)
print(type(val_iter))

images, labels = val_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

out = torchvision.utils.make_grid(images)

imshow(out, title=[labels])

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16*47*47, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 29)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.dropout2d(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

lenet = LeNet()

writer_train.add_graph(lenet, Variable((torch.Tensor(train_loader.dataset.dataImages[0:1])).cpu(), ))

if is_cuda:
    lenet.cuda()

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()

#criterion = F.cross_entropy()
optimizer = torch.optim.Adam(lenet.parameters(), lr=learning_rate)

def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):

        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)

        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)

        running_loss += criterion(output, target).cpu().data.item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        '{} loss is {} and {} accuracy is {}/{}, {}'.format(phase, loss, phase, running_correct, len(data_loader.dataset), accuracy ))
    return loss, accuracy

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

lenet.apply(init_weights)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

start = clock()
for epoch in range(0, epochs):
    epoch_loss, epoch_accuracy = fit(epoch, lenet, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, lenet, val_loader, phase='validation')

    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)

    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    writer_train.add_scalars("losses", {'train_ln':epoch_loss, 'val_ln':val_epoch_loss}, int(epoch))
    writer_train.add_scalars("accuracies", {'train_ln':epoch_accuracy, 'val_ln':val_epoch_accuracy}, int(epoch))

    scheduler.step(val_epoch_loss)

writer_train.add_histogram("error_ln", np.array(train_losses))

elapsed = clock() - start

print(elapsed)

classes = ['A',  'B',  'C',  'D',  'del',  'E',  'F',
           'G',  'H',  'I',  'J',  'K',  'L',  'M',  'N',
           'nothing',  'O',  'P',  'Q',  'R',  'S',  'space',
           'T',  'U',  'V',  'W',  'X',  'Y',  'Z']

correct = 0
total = 0

y_pred = np.empty([0])
y_true = np.empty([0])
y_score = np.empty([0])

lenet.eval()
for images, labels in val_loader:
    if is_cuda:
        images = images.cuda()
    images = Variable(images)
    outputs = lenet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    y_pred = np.concatenate((y_pred, predicted.data.cpu().numpy()), axis=None)
    y_true = np.concatenate((y_true, labels.data.cpu().numpy()), axis=None)
    y_score = np.concatenate((y_score, outputs.detach().cpu().numpy()), axis=None)


conf_matrix = metrics.confusion_matrix(y_true, y_pred)

np.trace(conf_matrix)

print("classification rate :", np.trace(conf_matrix)/870)

print('Accuracy of the network on the validation set : %d %%' % (100 * correct // total))

df_conf_matrix = pd.DataFrame(data=conf_matrix, index=classes, columns=classes)

print("Confusion Matrix for the 29 classes:", df_conf_matrix)

classes_accuracy = []
for i in range(len(classes)):
    acc = conf_matrix[i, i]/conf_matrix[i, :].sum()
    classes_accuracy.append(acc)

df_classes = pd.DataFrame(data=classes_accuracy, index=classes, columns=['Accuracy'])

pr, rc, f1score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')

y_true_binarized = label_binarize(y_true, classes=[i for i in range(29)])
classes_auc = []

for i in range(29):
    auc_c = metrics.roc_auc_score(y_true_binarized[:,i], y_score.reshape(870, 29)[:,i])
    classes_auc.append(auc_c)


df_classes['AUC']=classes_auc

df_classes.sort_values(by=['Accuracy'])

def plot_ROC(idx_class):
    plt.figure()
    lw = 2
    fpr, tpr, _ = metrics.roc_curve(y_true_binarized[:,idx_class], y_score.reshape(870, 29)[:,idx_class])
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class '+ str(idx_class))
    plt.legend(loc="lower right")
    plt.show()

plot_ROC(2)

error_std = np.array(val_losses).std()
error_mean = np.array(val_losses).mean()

print(error_std)
print(error_mean)
print(pr)
print(rc)
print(f1score)

torch.save(lenet.state_dict(), 'lenetModelfinal.pkl')
print("testingLeNetModel.py: completed")
