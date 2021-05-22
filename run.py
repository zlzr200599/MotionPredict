from MultiDataset import AllDataset
import warnings
from models.SandwichNet import SandwichNet
import sys
import getopt

model = SandwichNet(input_size=2,
                    en_layers=2, en_hidden_size=16,
                    att_layers=2, att_size=32, d_ff=64, n_head=8,
                    de_layers=2, output_size=2,
                    saved_path='model.pth'
                    )


# train_dir = '/home/huanghao/Lab/argodataset/train/data'
# val_dir='/home/huanghao/Lab/argodataset/val/data'
# test_dir = './dataset/test/test_2k'

try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["train", "test","traindir=", "valdir=","testdir="])
except:
    print("Error")
cmd = {opt: arg for opt, arg in opts}
train_dir, val_dir, test_dir = cmd['--traindir'], cmd['--valdir'], cmd['--testdir']
data = AllDataset(train_dir=train_dir,
                  train_fraction=0.2,
                  val_dir=val_dir,
                  val_fraction=500 / 39472,  # 39472
                  test_dir=test_dir,
                  test_fraction=1.,
                  )
print(cmd)
if "--train" in cmd:
    model.train_model(dataset=data, batch_size=16, shuffle=True,
                      n_epoch=30, lr=0.001,
                      )
if "--test" in cmd:
    model.test_model(dataset=data, output_dir="./test_output/")