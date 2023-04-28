

## RNN Question Answering Model using pytorch
import torch as th
from torch import nn
from utils import utils


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = th.nn.RNN(input_size, hidden_size)
        self.linear = th.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        h = th.zeros(1, self.hidden_size)
        out, hidden = self.rnn(x, h)
        out = self.linear(out)
        return out, hidden
    
        
    def init_hidden(self, batch_size=1):
        """
        Initializes hidden state, of zeros.
        """
        return th.zeros(batch_size, self.hidden_size)
        
    # Not working yet
    def train(self, input,loss_fn, optimizer, epochs, batch_size):
        print(f"Starting Training on {epochs} epochs with batch size {batch_size}")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            train_loader = th.utils.data.DataLoader(input, batch_size=batch_size, shuffle=True)
            for inputs, labels in train_loader:
                inputs = list(inputs)
                labels = list(labels)


                inputs = [utils.clean_text(quest) for quest in inputs]
                inputs = [utils.tokenize_text_words(quest) for quest in inputs]
                inputs = [[utils.word_embedding(tensor) for tensor in quest] for quest in inputs]


                labels = [utils.clean_text(quest) for quest in labels]
                labels = [utils.tokenize_text_words(quest) for quest in labels]
                labels = [[utils.word_embedding(tensor) for tensor in quest] for quest in labels]

        
                # Forward pass
                for idx in range(len(inputs)):
                    x = th.stack(inputs[idx])
                    y = th.stack(labels[idx])

                    output, _ = self(x)
                    print(output.shape)
                    print(y.shape)

                    # Calculate the loss
                    loss = loss_fn(output, y)

                    # Backpropagate the loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
            print("Epoch Loss: ", loss.item())
        print("Training Complete")
        return output, loss.item()

    # Not working yet
    def eval(self, input, batch_size):
        correct = 0
        total = 0
        train_loader = th.utils.data.DataLoader(input, batch_size=batch_size, shuffle=True)
        for inputs, labels in train_loader:
            inputs = list(inputs)
            labels = list(labels)


            inputs = [utils.clean_text(quest) for quest in inputs]
            inputs = [utils.tokenize_text_words(quest) for quest in inputs]
            inputs = [[utils.word_embedding(tensor) for tensor in quest] for quest in inputs]
            # inputs = torch.stack(inputs)


            labels = [utils.clean_text(quest) for quest in labels]
            labels = [utils.tokenize_text_words(quest) for quest in labels]
            labels = [[utils.word_embedding(tensor) for tensor in quest] for quest in labels]

        # Forward pass
        for idx in range(len(inputs)):
            x = th.stack(inputs[idx])
            y = th.stack(labels[idx])

            output = self(x)

            correct += (output == y).sum().item()
            accuracy = correct / batch_size
            print("Accuracy:", accuracy)

            
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
