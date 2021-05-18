import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.agent_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout)
        self.av_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, dropout=dropout)
        self.others_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                      num_layers=num_layers, dropout=dropout)
        self.lane_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, dropout=dropout)
        self.proj = nn.Linear(hidden_size, output_size,)
        self.num_layers = num_layers

    def forward(self, agent: torch.Tensor, av: torch.Tensor, others: torch.Tensor, lane: torch.Tensor):
        agent, av, others, lane = agent.transpose(0,1),av.transpose(0,1), others.transpose(0,1), lane.transpose(0,1)
        _, (agent_hn, agent_cn) = self.agent_encoder(agent)
        agent_code = self.proj(agent_hn[-1, :, :])  # agent_code shape: n_agent * output_size
        _, (av_hn, av_cn) = self.av_encoder(av)
        av_code = self.proj(av_hn[-1, :, :])
        _, (others_hn, others_cn) = self.others_encoder(others)
        others_code = self.proj(others_hn[-1, :, :])
        _, (lane_hn, lane_cn) = self.lane_encoder(lane)
        lane_code = self.proj(lane_hn[-1, :, :])
        return agent_code, av_code, others_code, lane_code




if __name__ == "__main__":
    model = LSTMEncoder(2, 64, 5, 3, 0.1)
    aa = torch.randn((6,20,2))
    bb = torch.randn((1,20,2))
    cc = torch.randn((2,15,2))
    dd = torch.randn((3,20,2))

    a,b,c,d = model(aa, bb, cc, dd)
    print(a)
    print(a.shape)
    print(c)
    print(c.shape)