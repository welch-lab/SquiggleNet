import time
import torch
import click
import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
from scipy import stats
from ont_fast5_api.fast5_interface import get_fast5_file
import pyslow5
from model import ResNet
from model import Bottleneck


########################
##### Normalization ####
########################
def normalization(data_test, batchi):
	mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
	m = np.median(data_test, axis=1)   
	data_test = ((data_test - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))

	x = np.where(np.abs(data_test) > 3.5)
	for i in range(x[0].shape[0]):
		if x[1][i] == 0:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]+1]
		elif x[1][i] == 2999:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]-1]
		else:
			data_test[x[0][i],x[1][i]] = (data_test[x[0][i],x[1][i]-1] + data_test[x[0][i],x[1][i]+1])/2

	data_test = torch.tensor(data_test).float()

	print("[Step 2]$$$$$$$$$$ Done data normalization with batch "+ str(batchi))
	return data_test


########################
####### Run Test #######
########################
def process(data_test, data_name, batchi, bmodel, outfile, device):
	with torch.no_grad():
		testx = data_test.to(device)
		outputs_test = bmodel(testx)
		with open(outfile + '/batch_' + str(batchi) + '.txt', 'w') as f:
			for nm, val in zip(data_name, outputs_test.max(dim=1).indices.int().data.cpu().numpy()):
				f.write(nm + '\t' + str(val)+'\n')
		print("[Step 3]$$$$$$$$$$ Done processing with batch "+ str(batchi))
		del outputs_test


########################
#### Load the data #####
########################
def get_raw_data(bmodel, batch, fileNM, data_test, data_name, cutoff, format, outfile, device):
	fast5_filepath = fileNM
	read_count = 0
	batchi = 0
	if format == 'fast5':
		with get_fast5_file(fast5_filepath, mode="r") as f5:
			for read in f5.get_reads():
				raw_data = read.get_raw_data(scale=True)
				append_data(read.read_id, raw_data, cutoff, data_test, data_name)
				read_count += 1
				if read_count == batch:
					if len(data_test) > 0:
						predict(data_test, data_name, batchi, bmodel, outfile, device)
						batchi += 1
						del data_test
						data_test = []
						del data_name
						data_name = []
					read_count = 0
	elif format == 'blow5':
		print(fast5_filepath)
		f5 = pyslow5.Open(fast5_filepath, 'r')
		for read in f5.seq_reads(pA=True):
			append_data(read['read_id'], read['signal'], cutoff, data_test, data_name)
			read_count += 1
			if read_count == batch:
				if len(data_test) > 0:
					predict(data_test, data_name, batchi, bmodel, outfile, device)
					batchi += 1
					del data_test
					data_test = []
					del data_name
					data_name = []
				read_count = 0
	else:
		print("Invalid raw data format!")
	return data_test, data_name


def append_data(read_id, raw_data, cutoff, data_test, data_name):
	if len(raw_data) >= (cutoff + 3000):
		data_test.append(raw_data[cutoff:(cutoff+3000)])
		data_name.append(read_id)




@click.command()
@click.option('--model', '-m', help='The pretrained model path and name', type=click.Path(exists=True))
@click.option('--infile', '-i', help='The input fast5 folder path', type=click.Path(exists=True))
@click.option('--outfile', '-o', help='The output result folder path', type=click.Path())
@click.option('--batch', '-b', default=1, help='Batch size')
@click.option('--cutoff', '-c', default=1500, help='Cutoff the first c signals')
@click.option('--format', '-ft', default='fast5', help='Raw file format (fast5, slow5)', type=str)
def main(model, infile, outfile, batch, cutoff, format):
	start_time = time.time()
	if torch.cuda.is_available:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Device: " + str(device))

	### make output folder
	if not os.path.exists(outfile):
		os.makedirs(outfile)

	### load model
	bmodel = ResNet(Bottleneck, [2,2,2,2]).to(device).eval()
	bmodel.load_state_dict(torch.load(model,  map_location=device))
	print("[Step 0]$$$$$$$$$$ Done loading model")


	### load data
	data_test = []
	data_name = []

	if format == 'slow5':
		format = 'blow5'

	if os.path.isdir(infile):
		infile = infile + '/*.' + format

	start_time = time.time()
	for fileNM in glob.glob(infile):
		data_test, data_name = get_raw_data(bmodel, batch, fileNM, data_test, data_name, cutoff, format, outfile, device)

	print("[Step FINAL]--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
	main()
