# Python to R

## Loading models saved in python

Currently the only way to load models from python is to rewrite the
model architecture in R. All the parameter names must be identical. A
complete example from Python to R is shown below. This is an extension
of the Serialization vignette.

An artificial neural net is implemented below in Python. Note the final
line which uses torch.save().

``` python
import torch
import numpy as np

#Make up data

madeUpData_x = np.random.rand(1000,100)
madeUpData_y = np.random.rand(1000)

#Convert to categorical
madeUpData_y = madeUpData_y.round()

train_py_X = torch.from_numpy(madeUpData_x).float()

train_py_Y = torch.from_numpy(madeUpData_y).float()

#Note that this class must be replicated identically in R
class simpleMLP(torch.nn.Module):
    def __init__(self):
        super(simpleMLP, self).__init__()
        self.modelFit = torch.nn.Sequential(
            torch.nn.Linear(100,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,1),
            torch.nn.Sigmoid())
            
    def forward(self, x):
        x =self.modelFit(x)

        return x

model = simpleMLP()


def modelTrainer(data_X,data_Y,model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):

        optimizer.zero_grad()

        yhat = model(data_X)

        loss = criterion(yhat,data_Y.unsqueeze(1))

        loss.backward()
        optimizer.step()

modelTrainer(data_X = train_py_X,data_Y = train_py_Y,model = model)

#-----------------------------------------------------------------
#save the model 

#Note that model.state_dict() comes out as an ordered dictionary
#The code below converts to a dictionary
stateDict = dict(model.state_dict())

#Note the argument _use_new_zipfile_serialization
torch.save(stateDict,f="path/babyTest.pth",
           _use_new_zipfile_serialization=True)
```

Once we have a saved .pth object we can load this into R. An example use
case would be training a model in Python then using Shiny to develop a
GUI for predictions from a trained model.

``` r
library(torch)

#Make up some test data
#note that proper installation of torch will yield no errors when we run
#this code
y <- torch_tensor(array(runif(8),dim = c(2,2,2)),dtype = torch_float64())

#Note the identical names between the Python class definition and our
#class definition
simpleMLP <- torch::nn_module(
  "simpleMLP",
  initialize = function(){
    
    self$modelFit <- nn_sequential(nn_linear(100,20),
                                   nn_relu(),
                                   nn_linear(20,1),
                                   nn_sigmoid())
    
  },
  forward = function(x){
    self$modelFit(x)
    }
)


model <- simpleMLP()

state_dict <- torch::load_state_dict("p/babyTest.pth")
model$load_state_dict(state_dict)

#Note that the dtype set in R has to match the made up data from Python
#More generally if reading new data into R you must ensure that it matches the 
#dtype that the model was trained with in Python
newData = torch_tensor(array(rnorm(n=1000),dim=c(10,100)),dtype=torch_float32())

predictMe <- model(newData)
```
