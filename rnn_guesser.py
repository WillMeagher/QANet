

## RNN Question Answering Model using pytorch
import torch as th
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """
        Initialize the model by setting up the various layers.
        """
        pass

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        pass

    def init_hidden(self, batch_size):
        """
        Initializes hidden state, of zeros.
        """
        pass
        
    def __call__(self, question, num_guesses=1) -> list[str]:
        """
        This function accepts a string representing a question and returns a list of tuples containing
        strings representing the guess.
        """
        pass

    def train(self, im_not_sure_what_to_put_here):
        """
        This function trains the model on the data for the given number of epochs.
        """
        pass

    def save(self, path: str):
        """
        This function saves the model to the path.
        """
        pass

    def load(self, path: str):
        """
        This function loads the model from the path.
        """
        pass
