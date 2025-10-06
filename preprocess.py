from ont_fast5_api.fast5_interface import get_fast5_file
import os
import glob
import click
import torch
import numpy as np
from scipy import stats

import random

from file_types.fast5_file import Fast5File, Fast5Read
from file_types.pod5_file import Pod5File, Pod5Read
from file_types.signal_file import SignalFile, SignalRead


def normalization(data_test, xi, outpath, pos = True, train = True):
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
	if pos is True:
		if train:
			torch.save(torch.tensor(data_test).float(), outpath + '/pos_' + str(xi) + '_train' + '.pt')
		else:
			torch.save(torch.tensor(data_test).float(), outpath + '/pos_' + str(xi) + '_val' + '.pt')
	else:
		if train:
			torch.save(torch.tensor(data_test).float(), outpath + '/neg_' + str(xi) + '_train' + '.pt')
		else:
			torch.save(torch.tensor(data_test).float(), outpath + '/neg_' + str(xi) + '_val' + '.pt')


def split_into_train_and_val(data, train, val, split_ratio):
	'''
		Splits the input data into training and validation sets by ratio defined by `split_ratio`. Note that selection for both sets is (pseudo)random.

		Provide empty (or training/validtion data from previous runs) lists for both train and val.

		`split_ratio` is the ratio of training data out of the entire dataset (rounded down to nearest integer).
		Hence, 1 - `split_ratio` gives the ratio of validation data out of the entire dataset.
	'''
	data_size = len(data)
	data_table = dict(zip([i for i in range(data_size)], data)) # hashtable makes search faster
	train_size = int(split_ratio*data_size)
	val_size = data_size - train_size
	val_table = dict(random.sample(list(data_table.items()), val_size))
	val.extend(list(val_table.values()))
	train.extend([v for k, v in data_table.items() if k not in val_table])
	assert len(train) + len(val) == len(data)


@click.command()
@click.option('--gtPos', '-gp', help='Ground truth list of positive read IDs')
@click.option('--gtNeg', '-gn', help='Ground trueh list of negative read IDs')
@click.option('--inpath', '-i', help='The input fast5 directory path')
@click.option('--outpath', '-o', help='The output pytorch tensor directory path')
@click.option('--batch', '-b', default=10000, help='Batch size, default 10000')
@click.option('--ratio', '-r', default=0.9, help='Ratio of reads for training per batch (rest is for validation, selected (pseudo)randomly)')
@click.option('--cutoff', '-c', default=1500, help='Cutoff the first c signals')

def main(gtpos, gtneg, inpath, outpath, batch, ratio, cutoff):
	### read in pos and neg ground truth variables
	my_file_pos = open(gtpos, "r")
	posli = my_file_pos.readlines()
	my_file_pos.close()
	posli = [pi.split('\n')[0] for pi in posli]

	my_file_neg = open(gtneg, "r")
	negli = my_file_neg.readlines()
	my_file_neg.close()
	negli = [pi.split('\n')[0] for pi in negli]

	### make output folder
	if not os.path.exists(outpath):
		os.makedirs(outpath)


	print("##### posli and negli length")
	print(len(posli))
	print(len(negli))
	print()

	### split fast5 files
	arrneg = []
	arrpos = []
	pi = 0
	ni = 0
	
	file_types = {
		'fast5': (Fast5File, Fast5Read),
		'pod5': (Pod5File, Pod5Read)
    }
	
    # This approach does allow for mixing of file types 
    # but that is almost always undesirable and impractical. It is
	# upto the user to ensure seperation of file types. This
	# code is only to ensure that the file types are automatically 
	# detected and handled without need for user intervention.
	files = []
	for ft in file_types:
		files.extend(glob.glob(inpath + f'/*.{ft}'))
	
	for fileNM in files:
		file_type = fileNM.split('.')[-1]
		FileClass, _ = file_types[file_type]
		with FileClass(fileNM) as f:
			print("##### file: " + fileNM)
			for read in f.get_reads():
				raw_data = read.get_raw_signal_pA()

				### only parse reads that are long enough
				if len(raw_data) >= (cutoff + 3000):
					if read.get_read_id() in posli:
						pi += 1
						arrpos.append(raw_data[cutoff:(cutoff + 3000)])
						if (pi%batch == 0) and (pi != 0):
							train = []
							val = []
							split_into_train_and_val(arrpos, train, val, ratio)
							# normalization(arrpos, pi, outpath, pos = True)
							normalization(train, pi, outpath, pos = True, train = True)
							normalization(val, pi, outpath, pos = True, train = False)
							del arrpos
							arrpos = []

					if read.get_read_id() in negli:
						ni += 1
						arrneg.append(raw_data[cutoff:(cutoff + 3000)])
						if (ni%batch == 0) and (ni != 0):
							train = []
							val = []
							split_into_train_and_val(arrneg, train, val, ratio)
							# normalization(arrneg, ni, outpath, pos = False)
							normalization(train, ni, outpath, pos = False, train = True)
							normalization(val, ni, outpath, pos = False, train = False)
							del arrneg
							arrneg = []


if __name__ == '__main__':
	main()
