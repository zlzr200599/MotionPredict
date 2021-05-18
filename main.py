from MultiDataset import AllDataset
import warnings
from models.SandwichNet import SandwichNet

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    train_dir = './dataset/train/train_10k'
    test_dir = './dataset/test/test_2k'

    data = AllDataset(train_dir=train_dir,
                      train_fraction=1.,
                      val_dir='/home/huanghao/Lab/argodataset/val/data',
                      val_fraction=500/39472,  # 39472
                      test_dir=test_dir,
                      test_fraction=1.,
                      )

    model = SandwichNet(input_size=2,
                        en_layers=1, en_hidden_size=16,
                        att_layers=1, att_size=64, d_ff=12, n_head=1,
                        de_layers=1,output_size=2,
                        saved_path='new_20210519_0000.pth'
                        )
    model.train_model(dataset=data, batch_size=8, shuffle=True,
                      n_epoch=100, lr=0.05,
                      )
    # model.val_model(dataset=data, return_to_plot=True)
    # model.test_model(dataset=data, output_dir="./test_result_0519_01/")
