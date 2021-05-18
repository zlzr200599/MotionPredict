import torch
import torch.nn as nn

from models.MultiAgentEncoder import LSTMEncoder
from models.MultiAgentDecoder import LSTMDecoder
from models.TransformerEncoder import make_model,Encoder
from torch.nn.utils.rnn import pad_sequence
import dgl


class SandwichNet(nn.Module):
    def __init__(self, input_size,
                 en_layers, en_hidden_size,
                 att_layers, att_size, d_ff, n_head,
                 de_layers, output_size=2,
                 dropout=0.0,):
        super(SandwichNet, self).__init__()
        self.lstm_encoder = LSTMEncoder(input_size=input_size, hidden_size=en_hidden_size,output_size=att_size,
                                        num_layers=en_layers, dropout=dropout)
        self.self_att: Encoder = make_model(n_layer=att_layers, d_model=att_size, d_ff=d_ff,
                                   n_head=n_head, dropout=dropout)
        self.lstm_decoder = LSTMDecoder(input_size=att_size, output_size=output_size,
                                        num_layers=de_layers,dropout=dropout)

    def forward(self, g: dgl.DGLGraph):
        # --------------------------ENCODER------------------------------------------------
        agent: torch.FloatTensor = g.nodes['agent'].data['state'][:, :20, :]
        av: torch.FloatTensor = g.nodes['av'].data['state']
        others: torch.FloatTensor = g.nodes['others'].data['state']
        lane: torch.FloatTensor = g.nodes['lane'].data['state']
        agent_code, av_code, others_code, lane_code = self.lstm_encoder(agent, av, others, lane)

        # ---------------------------BERT--------------------------------------------------
        g.nodes['agent'].data['code'] = agent_code
        g.nodes['av'].data['code'] = av_code
        g.nodes['others'].data['code'] = others_code
        g.nodes['lane'].data['code'] = lane_code

        all_lens, all_tensor = [], []
        for idx, cg in enumerate(dgl.unbatch(g)):
            c_agent = cg.nodes['agent'].data['code']
            c_av = cg.nodes['av'].data['code']
            c_others = cg.nodes['others'].data['code']
            c_lane = cg.nodes['lane'].data['code']

            len_list = [c_agent, c_av, c_others, c_lane]
            lines_tensor = torch.cat(len_list)
            lines_lens = [len(lines) for lines in len_list]
            all_lens.append(lines_lens)
            all_tensor.append(lines_tensor)
        # padded_tensor shape = [n_seq, n_batch, n_encoder_output_size]
        padded_tensor: torch.Tensor = pad_sequence(all_tensor)
        max_len, n_batch = padded_tensor.shape[0], padded_tensor.shape[1]
        mask = [[int(i < len(_s)) for i in range(max_len)] for _s in all_tensor]
        mask = torch.BoolTensor(mask)
        # print("mask\t",mask)
        padded_tensor = self.self_att(padded_tensor.transpose(0, 1), mask=mask)
        # print("padded tenser shape\t", padded_tensor.shape)
        # print("all lens\t", all_lens)
        padded_to_graph(pad_tensor=padded_tensor, all_len=all_lens, g=g)

        #  ---------------------decoder---------------------------------------
        agent: torch.FloatTensor = g.nodes['agent'].data['att']
        av: torch.FloatTensor = g.nodes['av'].data['att']
        others: torch.FloatTensor = g.nodes['others'].data['att']
        agent, av, others = self.lstm_decoder(agent, av, others)
        print('agent shape  ', agent.shape)
        print('av shape  ', av.shape)
        print('others shape  ', others.shape)
        return g

def padded_to_graph(pad_tensor, all_len, g: dgl.DGLGraph):
    agent, av, others, lane = [], [], [], []
    for n_row, e4 in enumerate(all_len):
        s, index = 0, []
        for i in e4:
            index.append((s, s+i))
            s += i
        # print(index)
        a = pad_tensor[n_row, index[0][0]:index[0][1], :]
        b = pad_tensor[n_row, index[1][0]:index[1][1], :]
        c = pad_tensor[n_row, index[2][0]:index[2][1], :]
        # d = pad_tensor[n_row, index[3][0]:index[3][1], :]
        agent.append(a)
        av.append(b)
        others.append(c)
        # lane.append(d)
    g.nodes['agent'].data['att'] = torch.cat(agent)
    g.nodes['av'].data['att'] = torch.cat(av)
    g.nodes['others'].data['att'] = torch.cat(others)





if __name__ == "__main__":
    model = SandwichNet(input_size=2,
                        en_layers=1, en_hidden_size=32,
                        att_layers=1, att_size=64, d_ff=128, n_head=1,
                        de_layers=1,)
    n_av, n_other, n_lane = 1, 3, 5
    graph1 = dgl.heterograph({
        ('agent', 'agent_env', 'env'): ([0], [0]),
        ('av', 'av_env', 'env'): (list(range(n_av)), [0] * n_av),
        ('others', 'others_env', 'env'): (list(range(n_other)), [0] * n_other),
        ('lane', 'lane_env', 'env'): (list(range(n_lane)), [0] * n_lane),
    })

    graph1.nodes['agent'].data['state'] = torch.randn((20, 2), dtype=torch.float).unsqueeze(dim=0)
    graph1.nodes['av'].data['state'] = torch.randn((n_av, 20, 2), dtype=torch.float)
    # graph.nodes['av'].data['mask'] = th.tensor(av_tracks_mask, dtype=th.float)
    graph1.nodes['others'].data['state'] = torch.rand((n_other, 20, 2), dtype=torch.float)
    # graph.nodes['others'].data['mask'] = th.tensor(other_tracks_mask, dtype=th.float)
    graph1.nodes['lane'].data['state'] = torch.rand((n_lane, 20, 2), dtype=torch.float)

    n_av, n_other, n_lane = 1, 2, 3
    graph2 = dgl.heterograph({
        ('agent', 'agent_env', 'env'): ([0], [0]),
        ('av', 'av_env', 'env'): (list(range(n_av)), [0] * n_av),
        ('others', 'others_env', 'env'): (list(range(n_other)), [0] * n_other),
        ('lane', 'lane_env', 'env'): (list(range(n_lane)), [0] * n_lane),
    })

    graph2.nodes['agent'].data['state'] = torch.randn((20, 2), dtype=torch.float).unsqueeze(dim=0)
    graph2.nodes['av'].data['state'] = torch.randn((n_av, 20, 2), dtype=torch.float)
    # graph.nodes['av'].data['mask'] = th.tensor(av_tracks_mask, dtype=th.float)
    graph2.nodes['others'].data['state'] = torch.rand((n_other, 20, 2), dtype=torch.float)
    # graph.nodes['others'].data['mask'] = th.tensor(other_tracks_mask, dtype=th.float)
    graph2.nodes['lane'].data['state'] = torch.rand((n_lane, 20, 2), dtype=torch.float)
    bgh = dgl.batch([graph1, graph2])

    model(bgh)
