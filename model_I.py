import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class StageNet_I(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(StageNet_I, self).__init__()
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels
        

        # new regular LSTM model
        self.lstm = nn.LSTM(input_dim, hidden_dim, bias=True, batch_first=True, dropout=dropout)
        
        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), stride=1, padding=4)
        self.nn_output = nn.Linear(int(self.conv_dim), int(output_dim), bias=True)
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)
    
    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    
    
    def forward(self, input, time, device):
        # batch_size, time_step, feature_dim = input.size()
        
        out, _ = self.lstm(input)
    
        if self.dropout > 0.0:
            out = self.nn_dropout(out)

        out = out.permute(0, 2, 1)
        out = self.nn_conv(out)
        out = torch.cat((torch.zeros(out.size(0), out.size(1), 1), out), -1)
        out = out.permute(0, 2, 1)
        out = self.nn_output(out)
        
        output = torch.sigmoid(out)


        return output
