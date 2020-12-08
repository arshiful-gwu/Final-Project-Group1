import numpy as np
from sklearn.model_selection import train_test_split

print("load_data_p3.py: Started")

imagesLoaded = np.load("file_val_test_Images.npy")
labelsLoaded = np.load("file_val_test_Labels.npy")

print("Splitting to form validation and test...")

X_val, X_test, y_val, y_test = train_test_split(imagesLoaded, labelsLoaded, test_size=.5, stratify=labelsLoaded, shuffle=True, random_state=68)

np.save('file_test_Labels', y_test)
np.save('file_val_Labels', y_val)

np.save('file_test_Images', X_test)
np.save('file_val_Images', X_val)

print("Validation and test dataset formed:")
print("file_test_Labels.npy")
print("file_test_Images.npy")
print("file_val_Labels.npy")
print("file_val_Images.npy")

print("load_data_p3.py: Completed")