import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
class ConBNRelu(nn.Module):
	def __init__(self, in_channels, out_channels, k_size, stride, padding):
		super(ConBNRelu, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, k_size, stride, padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		return self.block(x)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		vgg16 = models.vgg16_bn(pretrained=True)
		params = list(vgg16.features.children())
		self.enc_block1 = nn.Sequential(*params[:7])
		self.enc_block2 = nn.Sequential(*params[7:14])
		self.enc_block3 = nn.Sequential(*params[14:24])
		self.enc_block4 = nn.Sequential(*params[24:34])
		self.enc_block5 = nn.Sequential(*params[34:44])

		self.dec_block5 = nn.Sequential(
			ConBNRelu(512, 512, 3, 1, 1),
			ConBNRelu(512, 512, 3, 1, 1),
			ConBNRelu(512, 512, 3, 1, 1),
			nn.Dropout2d(0.5),
			nn.Upsample(scale_factor=2, mode='bilinear')
		)

		self.dec_block4 = nn.Sequential(
			ConBNRelu(512, 512, 3, 1, 1),
			ConBNRelu(512, 512, 3, 1, 1),
			ConBNRelu(512, 256, 3, 1, 1),
			nn.Dropout2d(0.5),
			nn.Upsample(scale_factor=2, mode='bilinear')
		)
		self.dec_block3 = nn.Sequential(
			ConBNRelu(256, 256, 3, 1, 1),
			ConBNRelu(256, 256, 3, 1, 1),
			ConBNRelu(256, 128, 3, 1, 1),
			nn.Dropout2d(0.5),
			nn.Upsample(scale_factor=2, mode='bilinear')
		)

		self.dec_block2 = nn.Sequential(
			ConBNRelu(128, 128, 3, 1, 1),
			ConBNRelu(128, 64, 3, 1, 1),
			nn.Upsample(scale_factor=2, mode='bilinear')
		)
		self.dec_block1 = nn.Sequential(
			ConBNRelu(64, 64, 3, 1, 1),
			ConBNRelu(64, 1, 1, 1, 0),
			nn.Sigmoid()
		)


		self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
		self.up_sample2 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.up_sample3 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.up_sample4 = nn.Upsample(scale_factor=2, mode='bilinear')
		self.up_sample5 = nn.Upsample(scale_factor=2, mode='bilinear')

		self.deconv4 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
		self.deconv3 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
		self.deconv2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
		self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
		#self.copy_params_from_vgg16()

	def forward(self, x):
		h1 = self.enc_block1(x)
		h2 = self.enc_block2(h1)
		h3 = self.enc_block3(h2)
		h4 = self.enc_block4(h3)
		h5 = self.enc_block5(h4)
		x = self.up_sample5(h5)


		x = self.dec_block5(x) + self.deconv4(h4)
		x = self.dec_block4(x) + self.deconv3(h3)
		x = self.dec_block3(x) + self.deconv2(h2)
		x = self.dec_block2(x) + self.deconv1(h1)
		x = self.dec_block1(x)
		return torch.squeeze(x)







if __name__ == "__main__":
	a = Net().cuda()
	#from torch.autograd import  Variable
	#d = Variable(torch.ones(4, 3, 256, 256)).cuda()
	#o = a(d)
	print a