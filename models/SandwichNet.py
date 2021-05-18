import torch
import torch.nn as nn
import os
import time
from MultiDataset import AllDataset,collate
from models.MultiAgentEncoder import LSTMEncoder
from models.MultiAgentDecoder import LSTMDecoder
from models.TransformerEncoder import make_model,Encoder
from torch.nn.utils.rnn import pad_sequence
import dgl
from dgl.dataloading import GraphDataLoader
from torch.optim.lr_scheduler import LambdaLR
from utils import val_plot
import pandas as pd
import numpy as np
from collections import deque

class SandwichNet(nn.Module):
    def __init__(self, input_size,
                 en_layers, en_hidden_size,
                 att_layers, att_size, d_ff, n_head,
                 de_layers, output_size=2,
                 dropout=0.0,
                 saved_path='new.pth'
                 ):
        super(SandwichNet, self).__init__()
        self.saved_path = saved_path
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
        padded_tensor = self.self_att(padded_tensor.transpose(0, 1), mask=mask) + padded_tensor.transpose(0, 1)
        # print("padded tenser shape\t", padded_tensor.shape)
        # print("all lens\t", all_lens)
        SandwichNet.padded_to_graph(pad_tensor=padded_tensor, all_len=all_lens, g=g)

        #  ---------------------decoder---------------------------------------
        agent: torch.FloatTensor = g.nodes['agent'].data['att']
        av: torch.FloatTensor = g.nodes['av'].data['att']
        others: torch.FloatTensor = g.nodes['others'].data['att']
        agent, av, others = self.lstm_decoder(agent, av, others)
        g.nodes['agent'].data['predict'] = agent
        g.nodes['av'].data['predict'] = av
        g.nodes['others'].data['predict'] = others
        # print('agent shape  ', agent.shape)
        # print('av shape  ', av.shape)
        # print('others shape  ', others.shape)
        return g

    def train_model(self, dataset: AllDataset, collate_fn=collate, batch_size=10, shuffle=True, drop_last=True,
                    n_epoch=10, lr=0.05,
                    ):
        self.training = True
        data_loader = GraphDataLoader(dataset.train, collate_fn=collate_fn, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=drop_last)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 0.9 ** e, verbose=True)
        start_time = time.time()
        loss_queue = deque(maxlen=10)
        real_queue = deque(maxlen=10)
        for epoch in range(n_epoch):
            for i, (bhg, info) in enumerate(data_loader):
                optimizer.zero_grad()
                self.forward(bhg)
                y_pred = bhg.nodes['agent'].data['predict']
                y_true = bhg.nodes['agent'].data['state'][:, 20:, :]

                x_pred = bhg.nodes['av'].data['predict'] * bhg.nodes['av'].data['mask'][:, 20:, :]
                x_true = bhg.nodes['av'].data['state'][:, 20:, :]
                # print(float(bhg.nodes['av'].data['mask'][:, 20:, :].mean()))

                z_pred = bhg.nodes['others'].data['predict'] * bhg.nodes['others'].data['mask'][:, 20:, :]
                z_true = bhg.nodes['others'].data['state'][:, 20:, :]
                # print(float(bhg.nodes['others'].data['mask'][:, 20:, :].mean()))
                pred, true = torch.cat((y_pred, x_pred, z_pred)),torch.cat((y_true, x_true, z_true)),
                # loss = criterion(y_pred, y_true)
                loss = criterion(pred, true)
                loss.backward()
                optimizer.step()

                loss_queue.append(loss)
                real_lose = torch.square(y_pred - y_true).flatten().view(-1, 2)
                real_lose = torch.sum(real_lose, dim=1)
                real_lose = torch.sqrt(real_lose)
                real_lose = torch.mean(real_lose)
                real_queue.append(real_lose)
                if i % 10 == 0:
                    print(
                        f"epoch: ({epoch}/{n_epoch}) | n_iter: ({i}/{len(data_loader)}) | "
                        f"loss : {sum(loss_queue) / len(loss_queue):6.4f} | "
                        f"{time.time() - start_time:6.2f} s "
                    )
                    print(f"real loss average error: {sum(real_queue) / len(real_queue):6.4f} m")
            scheduler.step()
            self.val_model(dataset=dataset)
            print("-------------------------------------------------------------------------------------------")
            print(f"have train: {epoch} epoch \n"
                  f"estimate time remain: {(n_epoch - epoch) * (time.time() - start_time)/(epoch+1):8.2f} s")
            print("-------------------------------------------------------------------------------------------")
            print(f"total time: {time.time() - start_time:6.2f} s")
        self.save()
        self.training = False

    @torch.no_grad()
    def val_model(self, dataset: AllDataset, return_to_plot=False):
        if not self.training:
            self.load()
        self.eval()
        data_loader = GraphDataLoader(dataset.val, collate_fn=collate,
                                      batch_size=int(10 if not return_to_plot else 1),
                                      shuffle=False, drop_last=False)
        start_time = time.time()
        real_queue = deque()
        for i, (bhg, info) in enumerate(data_loader):
            self.forward(bhg)
            agent_pred = bhg.nodes['agent'].data['predict']
            agent_true = bhg.nodes['agent'].data['state'][:, 20:, :]

            real_lose = torch.square(agent_pred - agent_true).flatten().view(-1, 2)
            real_lose = torch.sum(real_lose, dim=1)
            real_lose = torch.sqrt(real_lose)
            real_lose = torch.mean(real_lose)
            real_queue.append(real_lose)
            if return_to_plot:
                val_plot(bhg)
        print("-------------------------------------evaluation---------------------------------------------")
        print(f"val total time elapse: {time.time() - start_time:6.2f} s| #samples : {len(dataset.val)}"
              f" loss : {sum(real_queue) / len(real_queue):6.4f} m")
        print("--------------------------------------------------------------------------------------------")
        self.train()

    @torch.no_grad()
    def test_model(self, dataset: AllDataset, output_dir: str = "test_result"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"make new dir {os.path.abspath(output_dir)}, and write files into it.")
        else:
            print(f'output dir {os.path.abspath(output_dir)} exists !')
        self.load()
        self.eval()
        data_loader = GraphDataLoader(dataset.test, collate_fn=collate, batch_size=10,
                                      shuffle=False, drop_last=False)
        start_time = time.time()
        for i, (bhg, info) in enumerate(data_loader):
            batch_size = len(info)
            self.forward(bhg)
            y_pred: torch.FloatTensor = bhg.nodes['agent'].data['predict']
            assert batch_size == y_pred.shape[0]
            for n, d in enumerate(info):
                st = float(d['split_time'])
                x, y = d['radix']['x'], d['radix']['y']
                timestamp = pd.Series(np.linspace(st + 0.1, st + 3.0, 30, dtype=np.float), name="TIMESTAMP")
                track_id = pd.Series([d['agent_track_id'] for _ in range(30)], name="TRACK_ID")
                object_type = pd.Series(["AGENT" for _ in range(30)], name="OBJECT_TYPE")
                x = pd.Series(y_pred[n, :, 0] + x, name="X")
                y = pd.Series(y_pred[n, :, 1] + y, name="Y")
                city_name = pd.Series([d['city'] for _ in range(30)], name="CITY_NAME")
                this_df = pd.DataFrame(list(zip(timestamp, track_id, object_type, x, y, city_name)),
                                       columns=("TIMESTAMP", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "CITY_NAME")
                                       )
                stack_df = pd.concat(objs=[d['df'], this_df])

                stack_df.to_csv(os.path.join(output_dir, d['filename']+".csv"), index=False)

                # pd.set_option('display.max_columns', 1000)
                # print(this_df)
        self.train()
        print(f"test time is :{time.time() - start_time:6.2f} s | num_samples : {len(dataset.test)}")


    def save(self):
        torch.save(self.state_dict(), self.saved_path)
        print(f'save the model to {os.path.abspath(self.saved_path)}')
        return self.saved_path

    def load(self):
        self.load_state_dict(torch.load(self.saved_path))

    @staticmethod
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
                        att_layers=1, att_size=32, d_ff=32, n_head=1,
                        de_layers=1,dropout=0.0)
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
