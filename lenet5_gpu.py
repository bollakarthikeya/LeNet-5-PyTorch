
# coding: utf-8

# ### Implementing LeNet-5 Architecture On MNIST Dataset (GPU Implementation)

# In[1]:

import torch
torch.multiprocessing.set_start_method("spawn")        # https://github.com/pytorch/pytorch/issues/3491#event-1326332533
import torch.nn   
import torch.optim 
import torch.nn.functional 
import torchvision.datasets   
import torchvision.transforms     

from torch import np   # this is torch's wrapper for numpy 
import matplotlib
matplotlib.use('Agg')       
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot    
from matplotlib.pyplot import subplot     
from sklearn.metrics import accuracy_score


# In[2]:

# ---------- MNIST data from torch ----------     
# First download the dataset and set aside training and test data. Then perform transformation.  
# 'torchvision.transforms.compose()' creates a series of transformations to be applied on dataset. 
# 'torchvision' reads datasets into PILImage (Python imaging format) which are in [0, 255] range. 
# 'torchvision.transforms.ToTensor()' converts the PIL Image from range [0, 255] to a FloatTensor of 
# shape (C x H x W) with range [0.0, 1.0]
# We then renormalize the input of range [0, 1] to range [-1, 1] using μ = 0.5 and standard deviation = 0.5

# [Refer line 73] http://pytorch.org/docs/0.2.0/_modules/torchvision/datasets/mnist.html
# [Refer 'ToTensor' class] http://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html

transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
valid = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)  

# create training and validation set indexes (80-20 split)
idx = list(range(len(train)))
np.random.seed(1009)
np.random.shuffle(idx)          
train_idx = idx[ : int(0.8 * len(idx))]       
valid_idx = idx[int(0.8 * len(idx)) : ]


# In[3]:

# sample images
fig1 = train.train_data[0].numpy()  
fig2 = train.train_data[2500].numpy()
fig3 = train.train_data[25000].numpy()  
fig4 = train.train_data[59999].numpy()
subplot(2,2,1), pyplot.imshow(fig1)  
subplot(2,2,2), pyplot.imshow(fig2) 
subplot(2,2,3), pyplot.imshow(fig3)
subplot(2,2,4), pyplot.imshow(fig4)


# In[4]:

# generate training and validation set samples
train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)    
valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)  

# Load training and validation data based on above samples
# Size of an individual batch during training and validation is 30
# Both training and validation datasets are shuffled at every epoch by 'SubsetRandomSampler()'. Test set is not shuffled.
train_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=train_set, num_workers=4)  
valid_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=valid_set, num_workers=4)    
test_loader = torch.utils.data.DataLoader(test, num_workers=4)       


# In[5]:

# Defining the network (LeNet-5)  
class LeNet5(torch.nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))  
        # max-pooling with 2x2 grid 
        x = self.max_pool_1(x) 
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        
        return x
     
net = LeNet5()     
net.cuda()


# In[6]:

# set up loss function -- 'SVM Loss' a.k.a 'Cross-Entropy Loss'
loss_func = torch.nn.CrossEntropyLoss()
       
# SGD used for optimization, momentum update used as parameter update  
optimization = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)


# In[7]:

# Let training begin!
numEpochs = 20    
training_accuracy = []     
validation_accuracy = []

for epoch in range(numEpochs):
    
    # training set -- perform model training
    epoch_training_loss = 0.0
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function        
        # split training data into inputs and labels
        inputs, labels = training_batch                              # 'training_batch' is a list               
        # wrap data in 'Variable'
        inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())        
        # Make gradients zero for parameters 'W', 'b'
        optimization.zero_grad()         
        # forward, backward pass with parameter update
        forward_output = net(inputs)
        loss = loss_func(forward_output, labels)
        loss.backward()   
        optimization.step()     
        # calculating loss 
        epoch_training_loss += loss.data[0]
        num_batches += 1
        
    print("epoch: ", epoch, ", loss: ", epoch_training_loss/num_batches)            
     
    # calculate training set accuracy
    accuracy = 0.0 
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function        
        num_batches += 1
        inputs, actual_val = training_batch
        # perform classification
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function    
        predicted_val = predicted_val.cpu().data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
        predicted_val = np.argmax(predicted_val, axis = 1)  # retrieved max_values along every row    
        # accuracy   
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    training_accuracy.append(accuracy/num_batches)   

    # calculate validation set accuracy 
    accuracy = 0.0 
    num_batches = 0
    for batch_num, validation_batch in enumerate(valid_loader):        # 'enumerate' is a super helpful function        
        num_batches += 1
        inputs, actual_val = validation_batch
        # perform classification
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))    
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function    
        predicted_val = predicted_val.cpu().data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
        predicted_val = np.argmax(predicted_val, axis = 1)  # retrieved max_values along every row    
        # accuracy        
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    validation_accuracy.append(accuracy/num_batches)


# In[8]:

epochs = list(range(numEpochs))

# plotting training and validation accuracies
fig1 = pyplot.figure()
pyplot.plot(epochs, training_accuracy, 'r')
pyplot.plot(epochs, validation_accuracy, 'g')
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy") 
pyplot.show(fig1)


# In[9]:

# test the model on test dataset
correct = 0
total = 0
for test_data in test_loader:
    total += 1
    inputs, actual_val = test_data 
    # perform classification
    predicted_val = net(torch.autograd.Variable(inputs.cuda()))   
    # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
    predicted_val = predicted_val.cpu().data
    max_score, idx = torch.max(predicted_val, 1)
    # compare it with actual value and estimate accuracy
    correct += (idx == actual_val).sum()
       
print("Classifier Accuracy: ", correct/total * 100)

