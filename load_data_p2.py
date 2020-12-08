import numpy as np
from sklearn.model_selection import train_test_split

print("load_data_p2.py: Started")

imagesLoaded = np.load("ASLimages.npy")
labelsLoaded = np.load("ASLlabels.npy")

print("Splitting data into test and train...")

X_train, X_test, y_train, y_test = train_test_split(imagesLoaded, labelsLoaded, test_size=.3, stratify=labelsLoaded, shuffle=True, random_state=48)

np.save('file_val_test_Labels', y_test)
np.save('file_train_Labels', y_train)

np.save('file_val_test_Images', X_test)
np.save('file_train_Images', X_train)

print("Train and test dataset formed:")
print("file_train_Labels.npy")
print("file_train_Images.npy")
print("file_val_test_Labels.npy")
print("file_val_test_Images.npy")

print("load_data_p2.py: Completed")