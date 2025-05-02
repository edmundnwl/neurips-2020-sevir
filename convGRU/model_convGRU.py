import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                     out_channels=2*self.hidden_dim,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)

        self.conv_candidate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                        out_channels=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

    def forward(self, input_tensor, hidden_state):
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined_conv = self.conv_candidate(torch.cat([input_tensor, reset_gate*hidden_state], dim=1))
        candidate_state = torch.tanh(combined_conv)

        new_hidden_state = update_gate*hidden_state + (1-update_gate)*candidate_state

        return new_hidden_state

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=kernel_size,
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, c, h, w = input_tensor.size()

        # Initialize hidden state with zeros if not provided
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w),
                                             device=input_tensor.device)

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                               hidden_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        # Return the last time step output
        return cur_layer_input[:, -1, :, :, :], hidden_state

    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(torch.zeros(batch_size, self.hidden_dim[i], *image_size).to(device))
        return init_states

class ConvGRUNowcast(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=[64, 128], kernel_size=(3, 3), num_layers=2, output_dim=1):
        super(ConvGRUNowcast, self).__init__()
        self.convgru = ConvGRU(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                kernel_size=kernel_size,
                                num_layers=num_layers,
                                bias=True)
        self.conv = nn.Conv2d(in_channels=hidden_dim[-1],
                              out_channels=output_dim,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        x, _ = self.convgru(x)
        x = self.conv(x)
        return x