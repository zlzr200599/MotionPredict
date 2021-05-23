import os
import matplotlib.pyplot as plt
import math
import torch
import numpy as np
import pandas as pd
import dgl
from typing import List
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.evaluation.competition_util import generate_forecasting_h5


def batch_change_name(old_dir, new_dir):
    """
    test_dir = './test_result_0514'
    to_path = './开发者赛道-自动驾驶-凤凰于飞-1-2000/'

    """
    os.mkdir(new_dir)
    for i, f in enumerate(os.listdir(old_dir), 1):
        old_name = os.path.join(old_dir, f)
        new_name = os.path.join(new_dir, str(i) + '.csv')
        os.rename(old_name, new_name)
    print(os.listdir(new_dir))


def val_plot(bhg):
    agent = bhg.nodes['agent'].data['state'][0]
    predict_track = bhg.nodes['agent'].data['predict'][0]
    av = bhg.nodes['av'].data['state'].view(-1, 2)
    others = bhg.nodes['others'].data['state'].view(-1, 2)
    lane = bhg.nodes['lane'].data['state'].view(-1, 2)

    plt.figure(figsize=(10, 10))
    plt.plot(lane[:, 0], lane[:, 1], '.', color="gray")
    plt.plot(av[:, 0], av[:, 1], '.', color='black')
    plt.plot(others[:, 0], others[:, 1], 'r.')

    plt.plot(agent[:20, 0], agent[:20, 1], 'go')
    plt.plot(agent[20:, 0], agent[20:, 1], 'b.')
    plt.plot(predict_track[:, 0], predict_track[:, 1], 'y.')
    print(predict_track)
    plt.show()


def plot_bhg(bhg, plot_predict=False):
    agent = bhg.nodes['agent'].data['state'][0]
    predict_track = bhg.nodes['agent'].data['predict'][0]
    av = bhg.nodes['av'].data['state'].view(-1, 3)
    others = bhg.nodes['others'].data['state'].view(-1, 3)
    lane = bhg.nodes['lane'].data['state'].view(-1, 2)

    plt.figure(figsize=(10, 10))
    plt.plot(lane[:, 0], lane[:, 1], '.', color="gray")
    plt.plot(av[:, 1], av[:, 2], '.', color='black')
    plt.plot(others[:, 1], others[:, 2], 'r.')

    plt.plot(agent[:20, 1], agent[:20, 2], 'go')
    plt.plot(agent[20:, 1], agent[20:, 2], 'b.')
    if plot_predict:
        plt.plot(predict_track[:, 1], predict_track[:, 2], 'y.')
    plt.show()


def str_to_tensor(input_str: str, coding_len: int = 50, padding: str = '$') -> np.ndarray:
    """
    input_str = '00000000-0000-0000-0000-000000147400'
    [0,0,0,...,0]
    """
    assert (padding not in input_str) and (len(input_str) < coding_len)
    return np.array([ord(c) for c in input_str] + [ord(padding)] * (coding_len - len(input_str)), np.int)


def tensor_to_str(int_tensor: np.ndarray, padding='$') -> str:
    """
    :param padding:
    :param int_tensor:
    :return:
    """
    ord_list: list = int_tensor.tolist()
    return "".join([chr(int(i)) for i in ord_list]).strip(padding)


def graph_and_info_to_df_list(g: dgl.DGLGraph, d: dict, ) -> List[pd.DataFrame]:
    df_list = []
    x, y = d['radix']['x'], d['radix']['y']
    center = np.array([x, y])
    st = float(d['split_time'])
    timestamp_obs = np.array(d['timestamp'], dtype=np.float)
    timestamp_predict = np.linspace(st + 0.1, st + 3.0, 30, dtype=np.float)
    times = np.concatenate((timestamp_obs, timestamp_predict))
    city = d['city']
    track_id = tensor_to_str(g.nodes['agent'].data['track_id'][0].numpy())
    xy = torch.cat((g.nodes['agent'].data['state'][0], g.nodes['agent'].data['predict'][0])) + center
    assert len(xy) == 50
    df_list.append(six_in_df(times, track_id, 'AGENT', xy, city))

    av_traj = torch.cat((g.nodes['av'].data["state"][:, :20, :], g.nodes['av'].data["predict"]), 1) + center
    track_id = g.nodes['av'].data['track_id']
    le = g.nodes['av'].data['len']
    for i, (tr, ti, n) in enumerate(zip(av_traj, track_id, le)):
        if int(n[1]) == 20:
            ti = tensor_to_str(ti)
            df_list.append(six_in_df(times, ti, "AV", tr, city)[int(n[0]):])

    o_traj = torch.cat((g.nodes['others'].data["state"][:, :20, :], g.nodes['others'].data["predict"]), 1) + center
    track_id = g.nodes['others'].data['track_id']
    le = g.nodes['others'].data['len']
    for i, (tr, ti, n) in enumerate(zip(o_traj, track_id, le)):
        if int(n[1]) == 20:
            ti = tensor_to_str(ti)
            df_list.append(six_in_df(times, ti, "OTHERS", tr, city)[int(n[0]):])
    return df_list


def six_in_df(timestamp, track_id: str, object_type: str, xy, city_name: str):
    n_rows = len(timestamp)
    timestamp, track_id, object_type, x, y, city_name = pd.Series(timestamp), pd.Series([track_id] * n_rows), \
                                                        pd.Series([object_type] * n_rows), pd.Series(xy[:, 0]), \
                                                        pd.Series(xy[:, 1]), pd.Series([city_name] * n_rows)

    return pd.DataFrame(list(zip(timestamp, track_id, object_type, x, y, city_name)),
                        columns=["TIMESTAMP", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "CITY_NAME"])


def converter_csv_to_argo(input_path: str, output_path: str):
    afl = ArgoverseForecastingLoader(input_path)
    output_all = {}
    counter = 1
    for data in afl:
        print('\r' + str(counter) + '/' + str(len(afl)), end="")
        seq_id = int(data.current_seq.name[:-4])
        output_all[seq_id] = np.expand_dims(data.agent_traj[20:, :], 0)
        counter += 1
    generate_forecasting_h5(output_all, output_path)  # this might take awhile


if __name__ == "__main__":
    # batch_change_name('./test_output', './test_jd')

    converter_csv_to_argo(input_path="./JD_ALL_result/",
                          output_path="./eval_ai_h5/")

    # afl = ArgoverseForecastingLoader('/home/huanghao/Lab/argodataset/train/data')
    # c = 0
    # for data in afl:
    #     if len(data.agent_traj) < 50:
    #         print(data.current_seq)
    #         # raise  Exception
    #     print(f"\r{c}/{len(afl)} {c}", end='')
    #     c += 1