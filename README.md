# Classification-of-MNIST-Digits-Using-Pytorch

The goals of this assignment are as follows.
1. Understand how neural networks can be used for classification of images
2. Understand how a model can be created, trained, validated, and saved in PyTorch
3. Understand how to read/write and process images in python

## Classify Images using Fully Connected Neural Networks

In this part, you will create a complete neural network architecture in PyTorch consisting of multiple
layers. You are required to report results obtained from different experiments on the MNIST dataset.

![image](https://user-images.githubusercontent.com/105145104/168344671-f4405603-40db-4b17-aefd-251b7d5b3c29.png)

**MNIST Dataset:**

The dataset is attached with the assignment. This dataset contains 60000 training and 10000 test
samples. Each sample is a grayscale image of size 28x28. There are 10 classes in which each sample
is an image of one of the digits (0 to 9). Please note that in the MNIST dataset there are 10
categories. If we randomly guess one category, there is a 1/10 probability that it would be correct.
Therefore, you cannot (theoretically) make a classifier that performs worse than that. If you get
less than 10% accuracy in any of your experiments, you can safely assume that you are doing
something fundamentally wrong. 

![image](https://user-images.githubusercontent.com/105145104/168345962-8fc49ec2-001b-420a-a985-715ede5bfd82.png)

## Steps:

### 1. Data loading and normalization
Load MNIST images using loadDataset() function. Function should accept the path of the dataset,
size of training, validation and testing data, batch size and shuffle as input. Function should load training
and testing data and then select data samples using torch.utils.data.Subset() according to given training,
validation and test size. Function should return loaded training, validation and testing data. Function
should also normalize the data with zero mean and 0.5 variance. Figure out the images optimal resizing
parameters and load all the training and testing data with that size, zero mean per batch and 0.5
variance.

### 2. Initialize Network 
You have to create a function to initialize the network. Parameters should be entered by the user. If your
system has a GPU you can turn it on to improve performance. You can use pytorch built-in functions to
create and initialize the network.

`net = init_network(no_of_layers, input_dim, neurons_per_layer, dropout)`

neuron_per_layer should be a list having elements describing the number of neurons in each hidden
layer. Last element of the list should be the dimension of output (In our case 10). For example if you
pass following parameters to this function:

`net = init_network(2, 784, [100, 50, 10],0.2)`

It should return you the network architecture with parameters initialized:
```
Size of net(1).w = 784x100
Size of net(1).b = 100
Size of net(2).w = 100x50
Size of net(2).b = 50
Size of net(3).w = 50x10
Size of net(3).b = 10
```
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

### 3. Training 
Now create a function to train the initialized network. You also have to keep track of loss and accuracy
on training and validation data for each epoch to display loss and accuracy curves.

`net = train(net, train_set_x, train_set_y, valid_set_x, valid_set_y, learning_rate, training_epochs, loss_func, optimizer)`

For more information about dropout:
   1.This function returns a trained Neural Network net, loss_array, and accuracy_array
### 4. Save Network 
Write a function that saves your network so that it can be used later without training.
### 5. Load Network 
Write a function that loads your saved network so that it can be used without training. Function should
return loaded model.
### 6. Testing Step 
Now create a function that uses a trained network to make predictions on a test set.
pred = test(net, model, test_set_x, test_set_y)
Function should return predictions of model.
### 7. Visualize Results 
Write a function that plot loss and accuracy curves and sample images and prediction made by model
on them. Function should also plot confusion matrix, f1_score, and accuracy on test data. Review
sklearn.metrics for getting different metrics of predictions.
### 8. Main Function 
Now create a main function that calls all above functions in required order. Main function should accept
“Path of dataset”, “size of training data”, “size of validation data”, “size of testing data”, “number of
hidden layers”, “list having number of neurons in each layer”, “Loss function”, “optimizer”, “batch size”,
“learning rate”, “Is GPU enable(By default false if you do not have GPU)”, “drop out”, “Is training (default
is False)”, “Visualize Results (Default is False)” and “training epochs” as input. If “Is training” is true, the
function should start training from scratch, otherwise the function should load the saved model and
make predictions using it. While performing experiments you can use Google Colab or Jupyter
Notebook. Your code must print your name, roll no, your all best/default parameters, training, validation
and testing losses, accuracies, f1 scores and confusion matrix on test data and accuracy and loss curves
of training and validation data.
