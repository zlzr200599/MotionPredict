import os
import matplotlib.pyplot as plt
import math
import torch

def batch_change_name(old_dir, new_dir):
    """
    test_dir = './test_result_0514'
    to_path = './开发者赛道-自动驾驶-凤凰于飞-1-2000/'

    """
    os.mkdir(new_dir)
    for i, f in enumerate(os.listdir(old_dir), 1):
        old_name = os.path.join(old_dir, f)
        new_name = os.path.join(new_dir, str(i)+'.csv')
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


def str_to_tensor(input_str: str) -> torch.IntTensor:
    """
    >>> input_str = '00000000-0000-0000-0000-000000147400'
    >>> str_to_tensor(input_str)
    tensor([48, 48, 48, 48, 48, 48, 48, 48, 45, 48, 48, 48, 48, 45, 48, 48, 48, 48,
    45, 48, 48, 48, 48, 45, 48, 48, 48, 48, 48, 48, 49, 52, 55, 52, 48, 48],
    dtype=torch.int32)
    """
    return torch.IntTensor([ord(c) for c in input_str])


def tensor_to_str(int_tensor: torch.IntTensor) -> str:
    """

    input_tensor = tensor([48, 48, 48, 48, 48, 48, 48, 48, 45, 48, 48, 48, 48, 45, 48, 48, 48, 48,
        45, 48, 48, 48, 48, 45, 48, 48, 48, 48, 48, 48, 49, 52, 55, 52, 48, 48],
        dtype=torch.int32)
    >>> tensor_to_str(input_tensor)
    '00000000-0000-0000-0000-000000147400'
    """
    char_ord_list = int_tensor.tolist()
    return "".join([chr(int(i)) for i in char_ord_list])


if __name__ == "__main__":
    pass