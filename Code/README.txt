1. Open up pycharm
2. Make a new project (for example: final_project)
3. Import all the python files into the project 
4. On your pycharm : Go to tool > deployment > configuration and make sure you are connected to your GCP instance
5. On your pycharm : Go to File > settings > project and select Python Interpretor
6. Click on the gear button on the right side of python interpreter selection menu
7. Click on Add
8. Choose SSH interpreter
9. Choose Existing SSH Configuration and fix your mapping
10. Run the file load_data_p1.py on pycharm this will give you the following numpy files (ASLimages.npy, ASLlabels.npy) 
11. Run the file load_data_p2.py on pycharm this will give you the following numpy files (file_train_Labels.npy, file_train_Images.npy, file_val_test_Labels.npy, file_val_test_Images.npy)
12. Run the file load_data_p3.py on pycharm this will give you the following numpy files (file_test_Labels.npy, file_test_Images.npy, file_val_Labels.npy, file_val_Images.npy)
13. Run the following files to start the baseline model training:
- baselineModel.py
- finalBaselineModel.py
- testingBaselineModel.py
13. Run the following files to start the LeNet model training:
- leNetModel.py
- finalLeNetModel.py
- testingLenetModel.py
12. To see the loss and accuracy trend of Baseline Model run the following command in your MobaXterm:
tensorboard --logdir=finalrunsBMtest
13. Tensorboard will generate a link like (http://localhost:6006/)
14. Copy that generated link (not the above one)
15. Open another terminal of your GCP instance in MobaXterm and run the command : chromium-browser
16. Copy/ paste the tensorboard generated link in the browser to see the loss/accuracy trends

17. To see the loss and accuracy trend of LeNet Model run the following command in your MobaXterm:
tensorboard --logdir=finalrunsLNTesting
18. Tensorboard will generate a link like (http://localhost:6006/)
19. Copy that generated link (not the above one)
20. Open another terminal of your GCP instance in MobaXterm and run the command : chromium-browser
21. Copy/ paste the tensorboard generated link in the browser to see the loss/accuracy trends 