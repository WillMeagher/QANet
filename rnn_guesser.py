

## RNN Question Answering Model using pytorch
import torch as th
from torch import nn
from utils import utils


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        print("in init")
        """
        Initialize the model by setting up the various layers.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        print("in forward")
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        combined = th.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
        

    def init_hidden(self):
        print("in init_hidden")
        """
        Initializes hidden state, of zeros.
        """
        return th.zeros(1, self.hidden_size)
        
    # def __call__(self, question, num_guesses=1) -> list[str]:
    #     """
    #     This function accepts a string representing a question and returns a list of tuples containing
    #     strings representing the guess.
    #     """
    #     print("in __call__")
    #     pass

    def train(self, im_not_sure_what_to_put_here):
        """
        This function trains the model on the data for the given number of epochs.
        """
        print("in train")
        pass

    def save(self, path: str):
        """
        This function saves the model to the path.
        """
        print("in save")
        pass

    def load(self, path: str):
        """
        This function loads the model from the path.
        """
        print("in load")
        pass
