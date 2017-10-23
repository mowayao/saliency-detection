#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
from torch.optim.lr_scheduler import StepLR
import numpy as np
from data import DataFolder
from Net import Net
import time
import datetime
import os

data_dirs = [("/media/mowayao/data/salient/data/ECSSD/train/images", "/media/mowayao/data/salient/data/ECSSD/train/gt"),
             ("/media/mowayao/data/salient/data/MSRA10K/images", "/media/mowayao/data/salient/data/MSRA10K/gt"),
            ("/media/mowayao/data/salient/data/HKU-IS/imgs", "/media/mowayao/data/salient/data/HKU-IS/gt")
             ]

test_dirs = [("/media/mowayao/data/salient/data/ECSSD/test/images", "/media/mowayao/data/salient/data/ECSSD/test/gt")
]
def process_data_dir(data_dir):
	files = os.listdir(data_dir)
	files = map(lambda x: os.path.join(data_dir, x), files)
	return sorted(files)
DATA_DICT = {}

IMG_FILES = []
GT_FILES = []

IMG_FILES_TEST = []
GT_FILES_TEST = []

for dir_pair in data_dirs:
	X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
	IMG_FILES.extend(X)
	GT_FILES.extend(y)

for dir_pair in test_dirs:
	X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
	IMG_FILES_TEST.extend(X)
	GT_FILES_TEST.extend(y)

IMGS_train, GT_train = IMG_FILES, GT_FILES

train_folder = DataFolder(IMGS_train, GT_train, True)

train_data = DataLoader(train_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True)

test_folder = DataFolder(IMG_FILES_TEST, GT_FILES_TEST, False)
test_data = DataLoader(test_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)



date_time = datetime.datetime.fromtimestamp(
        int(time.time())
    ).strftime('%Y-%m-%d-%H-%M-%S')
net = Net().cuda()
optimizer = optim.Adam(
			[
			{'params': net.enc_block1.parameters(), 'lr': config.BASE_LEARNING_RATE},
	    	{'params': net.enc_block2.parameters(), 'lr': config.BASE_LEARNING_RATE},
	    	{'params': net.enc_block3.parameters(), 'lr': config.BASE_LEARNING_RATE},
			{'params': net.enc_block4.parameters(), 'lr': config.BASE_LEARNING_RATE},
			{'params': net.enc_block5.parameters(), 'lr': config.BASE_LEARNING_RATE},
			{'params': net.dec_block1.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.dec_block2.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.dec_block3.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.dec_block4.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.dec_block5.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.deconv1.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.deconv2.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.deconv3.parameters(), 'lr': config.LEARNING_RATE},
			{'params': net.deconv4.parameters(), 'lr': config.LEARNING_RATE},
			],
			lr=config.BASE_LEARNING_RATE, weight_decay=5e-4
			)

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

def criterion(logits, labels, is_weight=True):
	kernel_size = 15
	a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
	ind = a.ge(0.01) * a.le(0.99)
	ind = ind.float()
	weights = Variable(torch.tensor.torch.ones(a.size())).cuda()
	if is_weight:
		w0 = weights.sum()
		weights = weights + ind * 2
		w1 = weights.sum()
		weights = weights / w1 * w0
	return F.mse_loss(logits, labels), F.binary_cross_entropy(logits, labels)

def weight_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		nn.init.xavier_uniform(m.weight.data)
net.dec_block1.apply(weight_init)
net.dec_block2.apply(weight_init)
net.dec_block3.apply(weight_init)
net.dec_block4.apply(weight_init)
net.dec_block5.apply(weight_init)
net.deconv1.apply(weight_init)
net.deconv2.apply(weight_init)
net.deconv3.apply(weight_init)
net.deconv4.apply(weight_init)

#net.neterator.apply(weight_init)
#dis.apply(weight_init)
total_iter_cnt = 0
evaluation = nn.L1Loss()
best_eval = None
for epoch in xrange(1, config.NUM_EPOCHS+1):
	net.train()
	sum_train_mae = 0
	sum_train_loss = 0

	##train
	scheduler.step()
	for iter_cnt, (img_batch, label_batch, weights) in enumerate(train_data):
		total_iter_cnt += 1
		optimizer.zero_grad()
		weights = Variable(weights).cuda()
		img_batch = Variable(img_batch).cuda()
		label_batch = Variable(label_batch).cuda()
		pred_label = net(img_batch)
		bce_loss = F.binary_cross_entropy(pred_label, label_batch, weights)
		mae = evaluation(pred_label, label_batch)
		loss = bce_loss


		sum_train_loss += loss.data[0]
		sum_train_mae += mae.data[0]
		loss.backward()
		optimizer.step()

		print "Epoch:{}\t  {}/{}\t loss:{} \t mae:{}".format(epoch, iter_cnt+1,
		                                         len(train_folder)/config.BATCH_SIZE,
		                                         sum_train_loss/(iter_cnt+1),
		                                         sum_train_mae/(iter_cnt+1))
	##evaluate
	net.eval()
	sum_eval_mae = 0
	sum_eval_loss = 0
	num_eval = 0
	for iter_cnt, (img_batch, label_batch, weights) in enumerate(test_data):
		img_batch = Variable(img_batch).cuda()
		label_batch = Variable(label_batch).cuda()
		pred_label = net(img_batch)
		probs = torch.squeeze(pred_label, dim=1)
		mae = evaluation(pred_label, label_batch)
		sum_eval_mae += mae.data[0] * img_batch.size(0)
		num_eval += img_batch.size(0)
	eval_mae = sum_eval_mae / num_eval
	print "Validation \t loss:{} \t mae:{}".format(epoch,
	                                         #eval_loss,
	                                         eval_mae)


	if best_eval is None or best_eval < eval_mae:
		best_eval = eval_mae
		state = {
			'net': net._modules,
			'mae': best_eval,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/best_model.pth')
	scheduler.step()
