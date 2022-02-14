import torch.nn as nn
from torch.autograd import Variable
import torch
from torch_geometric.nn import GCNConv, GATConv, GraphConv
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from .algorithm_utils import Algorithm, PyTorchUtils


class GCNLSTMCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, bias=True, seed: int=0, gpu: int=None):
        """
        Initialize GCNLSTM cell.
        
        Parameters
        ----------
        nodes_num: input
            Number of nodes.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(GCNLSTMCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias
        
        self.gconv = GCNConv(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=4 * self.hidden_dim,
                             bias=self.bias,
                             improved = True)

    def forward(self, input_tensor, cur_state, edge_index):
        '''
        input_tensor:(b,n,i)
        cur_state:[(b,n,h),(b,n,h)]
        '''
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along hidden axis
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])
        
        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # output: (b,n,h),(b,n,h)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))),
                self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))))

class GATLSTMCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, head=1, dropout=0, bias=True, seed: int=0, gpu: int=None):
        """
        Initialize GCNLSTM cell.
        
        Parameters
        ----------
        nodes_num: input
            Number of nodes.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        head: int
            Number of multi-head-attentions.
        dropout: float
            Dropout probability of the normalized attention coefficients.
        bias: bool
            Whether or not to add the bias.
        """

        super(GATLSTMCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.head = head
        self.dropout = dropout
        self.bias = bias
        
        self.gconv = GATConv(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=4 * self.hidden_dim,
                             heads=self.head,
                             concat = False,
                             dropout=self.dropout,
                             bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        '''
        input_tensor:(b,n,i)
        cur_state:[(b,n,h),(b,n,h)]
        '''
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along hidden axis
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])
        
        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # output: (b,n,h),(b,n,h)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))),
                self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))))

class WL1LSTMCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, bias=True, seed: int=0, gpu: int=None):
        """
        Initialize GCNLSTM cell.
        
        Parameters
        ----------
        nodes_num: input
            Number of nodes.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(WL1LSTMCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias
        
        self.gconv = GraphConv(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               aggr = 'mean',
                               bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        '''
        input_tensor:(b,n,i)
        cur_state:[(b,n,h),(b,n,h)]
        '''
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along hidden axis
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])
        
        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # output: (b,n,h),(b,n,h)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))),
                self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))))

class LSTMCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, bias=True, seed: int=0, gpu: int=None):
        """
        Initialize GCNLSTM cell.
        
        Parameters
        ----------
        nodes_num: input
            Number of nodes.
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSTMCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias
        
        self.gconv = nn.Linear(self.input_dim + self.hidden_dim,
                               4 * self.hidden_dim,
                               bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        '''
        input_tensor:(b,n,i)
        cur_state:[(b,n,h),(b,n,h)]
        '''
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=2)  # concatenate along hidden axis
        #batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])
        
        combined_conv = self.gconv(combined)
        #combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=2) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # output: (b,n,h),(b,n,h)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))),
                self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim))))

class GraphLSTM(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, num_layers, head=1, dropout=0, kind='GCN',
                 batch_first=False, bias=True, return_all_layers=True, seed: int=0, gpu: int=None):
        super(GraphLSTM, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        # Make sure that `hidden_dim` are lists having len == num_layers
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        head = self._extend_for_multilayer(head, num_layers)
        if not len(hidden_dim) == len(head) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head = head
        self.dropout = dropout
        self.kind = kind
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            if self.kind == 'GCN':
                cell_list.append(GCNLSTMCell(nodes_num=nodes_num,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             bias=self.bias,
                                             seed=self.seed,
                                             gpu=self.gpu))
            elif self.kind == 'GAT':
                cell_list.append(GATLSTMCell(nodes_num=nodes_num,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             head=self.head[i],
                                             dropout=self.dropout,
                                             bias=self.bias,
                                             seed=self.seed,
                                             gpu=self.gpu))
            elif self.kind == 'WL1':
                cell_list.append(WL1LSTMCell(nodes_num=nodes_num,
                                             input_dim=cur_input_dim,
                                             hidden_dim=self.hidden_dim[i],
                                             bias=self.bias,
                                             seed=self.seed,
                                             gpu=self.gpu))
            elif self.kind == 'LIN':
                cell_list.append(LSTMCell(nodes_num=nodes_num,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          bias=self.bias,
                                          seed=self.seed,
                                          gpu=self.gpu))
            else:
                raise NotImplementedError()

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, edge_index, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: 4-D Tensor either of shape (t, b, n, h) or (b, t, n, h)
        hidden_state: list [[(b, n, h), (b, n, h)]] * num_layers
            
        Returns
        -------
        last_output_list: [(b, t, n, h)] * num_layers(also determined by return_all_layers),
        layer_state_list: [[(b, n, h), (b, n, h)]] * num_layers(also determined by return_all_layers)
        """
        #确保(t, b, n, c)
        #if self.batch_first:
        #写在前面了
            # (b, t, n, c) -> (t, b, n, c)
            #input_tensor = input_tensor.permute(1, 0, 2, 3).contiguous()

        # Implement stateful GraphLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            # [[(b, n, h), (b, n, h)]] * num_layers
            hidden_state = self._init_hidden(input_tensor.size(1))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t],
                                                 edge_index = edge_index, cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param