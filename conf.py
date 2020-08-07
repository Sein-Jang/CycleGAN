from collections import defaultdict

config = defaultdict(dict)

TRAIN = dict()
VALID = dict()

TRAIN['lr'] = 2e-4

TRAIN['batch_size'] = 1
VALID['batch_size'] = 1

TRAIN['n_epoch'] = 200

TRAIN['beta_1'] = 0.5

TRAIN['dataset'] = 'dataset/vangogh2photo'
TRAIN['A_img_path'] = 'trainA/*.jpg'
TRAIN['B_img_path'] = 'trainB/*.jpg'

TRAIN['load_size'] = 286
TRAIN['crop_size'] = 256
