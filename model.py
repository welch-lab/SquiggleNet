import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F

def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
  return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, 
				   padding=padding, bias=False, dilation=padding, groups=groups)

def conv1(in_channel, out_channel, stride=1, padding=0):
  return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, 
				   padding=padding, bias=False)

def bcnorm(channel):
  return nn.BatchNorm1d(channel)


class Bottleneck(nn.Module):
	expansion = 1.5
	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1(in_channel, in_channel)
		self.bn1 = bcnorm(in_channel)
		self.conv2 = conv3(in_channel, in_channel, stride)
		self.bn2 = bcnorm(in_channel)
		self.conv3 = conv1(in_channel, out_channel)
		self.bn3 = bcnorm(out_channel)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
  
	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=2):
		super(ResNet, self).__init__()
		self.chan1 = 20

		# first block
		self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
		self.bn1 = bcnorm(self.chan1)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool1d(2, padding=1, stride=2)
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Linear(67, 2)

		self.layer1 = self._make_layer(block, 20, layers[0])
		self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 67, layers[3], stride=2)
		#self.layer5 = self._make_layer(block, 100, layers[4], stride=2)

		# initialization
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm1d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	
	def _make_layer(self, block, channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.chan1 != channels:
			downsample = nn.Sequential(
				conv1(self.chan1, channels, stride),
				bcnorm(channels),
			)

		layers = []
		layers.append(block(self.chan1, channels, stride, downsample))
		if stride != 1 or self.chan1 != channels:
		  self.chan1 = channels
		for _ in range(1, blocks):
			layers.append(block(self.chan1, channels))

		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		x = x.unsqueeze(1)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		#x = self.layer5(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		return x

	def forward(self, x):
	  return self._forward_impl(x)