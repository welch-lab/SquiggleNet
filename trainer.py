import os
import click
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from dataset import Dataset
from model import ResNet
from model import Bottleneck


@click.command()
@click.option('--tTrain', '-tt', help='The path of target sequence training set', type=click.Path(exists=True))
@click.option('--tVal', '-tv', help='The path of target sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of non-target sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of non-target sequence validation set', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output path and name for the best trained model')
@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', 
																type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=20, help='Number of epoches, default 20')
@click.option('--learningrate', '-l', default=1e-3, help='Learning rate, default 1e-3')

def main(ttrain, tval, ntrain, nval, outpath, interm, batch, epoch, learningrate):
	if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	# Parameters
	params = {'batch_size': batch,
				'shuffle': True,
				'num_workers': 10}

	### load files
	training_set = Dataset(ttrain, ntrain)
	training_generator = DataLoader(training_set, **params)

	validation_set = Dataset(tval, nval)
	validation_generator = DataLoader(validation_set, **params)

	zymo_train = torch.load(ttrain)
	hela_train = torch.load(ntrain)

	zymo_val = torch.load(tval)
	hela_val = torch.load(nval)

	### load model
	model = ResNet(Bottleneck, [2,2,2,2]).to(device)
	if interm is not None:
		model.load_state_dict(torch.load(interm))

	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

	bestacc = 0
	bestmd = None
	i = 0

	### Training
	for epoch in range(epoch):
		for spx, spy in training_generator:
				spx, spy = spx.to(device), spy.to(torch.long).to(device)

				# Forward pass
				outputs = model(spx)
				loss = criterion(outputs, spy)
				acc = 100.0 * (spy == outputs.max(dim=1).indices).float().mean().item()

				# Validation
				with torch.set_grad_enabled(False):
						acc_vt = 0
						vti = 0
						for valx, valy in validation_generator:
								valx, valy = valx.to(device), valy.to(device)
								outputs_val = model(valx)
								acc_v = 100.0 * (valy == outputs_val.max(dim=1).indices).float().mean().item()
								vti += 1
								acc_vt += acc_v
						acc_vt = acc_vt / vti
						if bestacc < acc_vt:
								bestacc = acc_vt
								bestmd = model
								torch.save(bestmd.state_dict(), outpath)
				
						print("epoch: " + str(epoch) + ", i: " + str(i) + ", bestacc: " + str(bestacc))
						i += 1
				
				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()



if __name__ == '__main__':
	main()
