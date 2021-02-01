#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

from .MLutils import ALL_LETTERS, N_LETTERS
from .MLutils import load_data, letter_to_tensor, line_to_tensor, random_training_example


class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    # # number of possible letters, hidden size, categories number
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        #Define 2 different liner layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        # combine input and hidden tensor
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        # apply the Linear layers and the softmax
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        # return 2 different tensors
        return output, hidden
    
    # need some initial hidden states in the begining.
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
# dictionary with the country as the key and names as values
category_lines, all_categories = load_data()
# number of categories
n_categories = len(all_categories)

# a Hyperparameter 
n_hidden = 128
# number of possible letters, hidden size, output size
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# one step
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
#print(output.size())
#>>> size: [1,18]
#print(next_hidden.size())
#>>> size: [1,128]
# whole sequence/name
input_tensor = line_to_tensor('if')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
print(output.size())
print(next_hidden.size())




# apply softmax in the end.
# this is the likelyhood of each character of each category
def category_from_output(output):
    # return index of the greatest value
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]
#print(category_from_output(output))




criterion = nn.NLLLoss()
# hyperparameter
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# whole name as tensor,
def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    #line_tensor.size()[0]: the length of the name
    for i in range(line_tensor.size()[0]):
        # apply the current character and the previous hidden state.
        output, hidden = rnn(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 100, 500
n_iters = 2000
loss = 0
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
        
    
# plt.figure()
# plt.plot(all_losses)
# plt.show()
# model can be saved

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        return(guess+" "+str(loss))




if __name__ == "__main__":
    predict("abcde 1 ifelse")

# In[ ]:




