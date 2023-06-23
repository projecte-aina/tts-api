# import the modules used here in this recipe
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time
import timeit

# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping ``nn.LSTM``, one layer, no preprocessing or postprocessing
# inspired by
# `Sequence Models and Long Short-Term Memory Networks tutorial <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html`_, by Robert Guthrie
# and `Dynamic Quanitzation tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.
class lstm_for_demonstration(nn.Module):
  """Elementary Long Short Term Memory style model which simply wraps ``nn.LSTM``
     Not to be used for anything other than demonstration.
  """
  def __init__(self,in_dim,out_dim,depth):
     super(lstm_for_demonstration,self).__init__()
     self.lstm = nn.LSTM(in_dim,out_dim,depth)

  def forward(self,inputs,hidden):
     out,hidden = self.lstm(inputs,hidden)
     return out, hidden


torch.manual_seed(29592)  # set the seed for reproducibility

#shape parameters
model_dimension=8
sequence_length=20
batch_size=1
lstm_depth=1

# random data for input
inputs = torch.randn(sequence_length,batch_size,model_dimension)
# hidden is actually is a tuple of the initial hidden state and the initial cell state
hidden = (torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension))

 # here is our floating point instance
float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)

# this is the call that does the work
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# show the changes that were made
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(float_lstm,"fp32")
q=print_size_of_model(quantized_lstm,"int8")
print("{0:.2f} times smaller".format(f/q))

# compare the performance
print("Floating point FP32")

# timeit.timer(float_lstm.forward(inputs, hidden))

start_time = timeit.default_timer()
float_lstm.forward(inputs, hidden)
end_time = timeit.default_timer()

execution_time = end_time - start_time

print(f"The non-quantized function took {execution_time} seconds to execute.")

print("Quantized INT8")

start_time = timeit.default_timer()
quantized_lstm.forward(inputs,hidden)
end_time = timeit.default_timer()

execution_time_quantized = end_time - start_time

print(f"The quantized function took {execution_time_quantized} seconds to execute.")

# timeit.timer(quantized_lstm.forward(inputs,hidden))

# run the float model
out1, hidden1 = float_lstm(inputs, hidden)
mag1 = torch.mean(abs(out1)).item()
print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

# run the quantized model
out2, hidden2 = quantized_lstm(inputs, hidden)
mag2 = torch.mean(abs(out2)).item()
print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

# compare them
mag3 = torch.mean(abs(out1-out2)).item()
print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3,mag3/mag1*100))