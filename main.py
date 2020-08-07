import glob
import os

from conf import *
from data import *
from model import generator, discriminator
from train import Trainer

print(tf.__version__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, TRAIN['dataset'])

"""
Configuration
"""
train_A_img_path = TRAIN['A_img_path']
train_B_img_path = TRAIN['B_img_path']
load_size = TRAIN['load_size']
crop_size = TRAIN['crop_size']
n_epoch = TRAIN['n_epoch']
beta_1 = TRAIN['beta_1']


"""
Load train data set
"""
train_A_img_path = glob.glob(os.path.join(DATASET, train_A_img_path))
train_B_img_path = glob.glob(os.path.join(DATASET, train_B_img_path))


"""
Train the model
"""
train_data_loader = make_dataset(train_A_img_path, train_B_img_path, load_size=load_size, crop_size=crop_size)
train_dataset = train_data_loader.dataset(TRAIN['batch_size'], random_transform=True, repeat_count=None)

trainer = Trainer(generator=generator(), discriminator=discriminator(),
                  cycle_loss_weight=10.0, identity_loss_weight=0.0, gradient_penalty_weight=10.0, beta_1=beta_1)
trainer.train(train_dataset, steps=n_epoch)


"""
Save the model weight
"""
trainer.A2B_G.save_weights('./weights/A2B_G.h5')
trainer.B2A_G.save_weights('./weights/B2A_G.h5')
trainer.A_D.save_weights('./weights/A_D.h5')
trainer.B_D.save_weights('./weights/B_D.h5')