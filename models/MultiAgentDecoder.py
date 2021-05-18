import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout):
        super(LSTMDecoder, self).__init__()
        dummy_t_size = 1
        self.agent_decoder = nn.LSTM(input_size=dummy_t_size, hidden_size=input_size,
                                     num_layers=num_layers, dropout=dropout)
        self.av_decoder = nn.LSTM(input_size=dummy_t_size, hidden_size=input_size,
                                  num_layers=num_layers, dropout=dropout)
        self.others_decoder = nn.LSTM(input_size=dummy_t_size, hidden_size=input_size,
                                      num_layers=num_layers, dropout=dropout)

        self.proj = nn.Linear(input_size, output_size)

        self.num_layers = num_layers

    def forward(self, agent: torch.Tensor, av: torch.Tensor, others: torch.Tensor):
        agent, n_agent = agent.unsqueeze(0).repeat(self.num_layers, 1, 1), len(agent)
        av, n_av= av.unsqueeze(0).repeat(self.num_layers, 1, 1), len(av)
        others, n_others = others.unsqueeze(0).repeat(self.num_layers, 1, 1), len(others)

        agent_t = torch.arange(0.0, 3.0, 0.1).unsqueeze(1).unsqueeze(0).expand(n_agent, -1, -1).permute(1, 0, 2)
        av_t = torch.arange(0.0, 3.0, 0.1).unsqueeze(1).unsqueeze(0).expand(n_av, -1, -1).permute(1, 0, 2)
        others_t = torch.arange(0.0, 3.0, 0.1).unsqueeze(1).unsqueeze(0).expand(n_others, -1, -1).permute(1, 0, 2)

        agent_track, _ = self.agent_decoder(agent_t, (agent, agent))
        agent_track = self.proj(agent_track)  # agent_code shape: n_agent * output_size
        av_track, _ = self.av_decoder(av_t, (av, av))
        av_track = self.proj(av_track)
        other_track, _ = self.others_decoder(others_t, (others, others))
        others_track = self.proj(other_track)

        return agent_track.transpose(0, 1), av_track.transpose(0, 1), others_track.transpose(0, 1)


if __name__ == "__main__":

    from models.MultiAgentEncoder import LSTMEncoder
    encoder = LSTMEncoder(2, 64, 5, 3, 0.1)
    aa = torch.randn((1,20,2))
    bb = torch.randn((1,20,2))
    cc = torch.randn((5,15,2))
    dd = torch.randn((3,20,2))

    a,b,c,d = encoder(aa, bb, cc, dd)

    decoder = LSTMDecoder(5, 2, 3, 0.1)
    x,y,z = decoder(a,b,c)

    print(x.shape,y.shape,z.shape)