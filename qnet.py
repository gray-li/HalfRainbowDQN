from torch import nn 
import torch.nn.functional as F 


class DeepQNetwork(nn.Module):

    def __init__(self, input_dims, output_dims, hidden_dims, lr):
        super().__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.output_dims = output_dims 
        self.hidden_dims = hidden_dims

        layer_dims = zip(hidden_dims[:-1], hidden_dims[1:])
        self.hidden = nn.ModuleList([nn.Linear(input_dims, hidden_dims[0])])
        self.hidden.extend([nn.Linear(h_in, h_out) for h_in, h_out in layer_dims])
        self.output = nn.Linear(hidden_dims[-1], self.output_dims)


    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))

        return self.output(x)

        
        
if __name__ == '__main__':
    dqn = DeepQNetwork((3), (4), [4,5,6], 0.01)
    dqn

