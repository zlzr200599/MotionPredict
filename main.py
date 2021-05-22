from MultiDataset import AllDataset
import warnings
from models.SandwichNet import SandwichNet

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    train_dir = '/home/huanghao/Lab/argodataset/train/data'
    # test_dir = './dataset/test/test_2k'
    test_dir = './dataset/test/test_all'

    data = AllDataset(train_dir=train_dir,
                      train_fraction=0.2,
                      val_dir='/home/huanghao/Lab/argodataset/val/data',
                      val_fraction=500 / 39472,  # 39472
                      test_dir=test_dir,
                      test_fraction=1.,
                      )

    # # model 0520  5000 loss : 2.5847 m
    # model = SandwichNet(input_size=2,
    #                     en_layers=2, en_hidden_size=32,
    #                     att_layers=2, att_size=64, d_ff=32, n_head=4,
    #                     de_layers=2,output_size=2,
    #                     saved_path='new_20210519_0000.pth'
    #                     )

    # # model 0521  5000 loss : 2.4547 m
    model = SandwichNet(input_size=2,
                        en_layers=2, en_hidden_size=16,
                        att_layers=2, att_size=32, d_ff=64, n_head=8,
                        de_layers=2, output_size=2,
                        saved_path='new_20210521.pth'
                        )

    # model.train_model(dataset=data, batch_size=16, shuffle=True,
    #                   n_epoch=30, lr=0.001,
    #                   )
    # model.val_model(dataset=data, return_to_plot=False)
    model.test_model(dataset=data, output_dir="./JD_ALL_result/")
    # model.test_all(dataset=data, output_dir="./test_result_all_agent_0520/")
