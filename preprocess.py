from ont_fast5_api.fast5_interface import get_fast5_file
import os
import glob
import click
import torch
import numpy as np
from scipy import stats


def normalization(data_test, xi, outpath, pos = True):
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
		torch.save(torch.tensor(data_test).float(), outpath + '/pos_' + str(xi) + '.pt')
	else:
		torch.save(torch.tensor(data_test).float(), outpath + '/neg_' + str(xi) + '.pt')

@click.command()
@click.option('--gtPos', '-gp', help='Ground truth list of positive read IDs')
@click.option('--gtNeg', '-gn', help='Ground trueh list of negative read IDs')
@click.option('--inpath', '-i', help='The input fast5 directory path')
@click.option('--outpath', '-o', help='The output pytorch tensor directory path')
@click.option('--batch', '-b', default=10000, help='Batch size, default 10000')
@click.option('--cutoff', '-c', default=1500, help='Cutoff the first c signals')

def main(gtpos, gtneg, inpath, outpath, batch, cutoff):
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
	
	for fileNM in glob.glob(inpath + '/*.fast5'):
		with get_fast5_file(fileNM, mode="r") as f5:
			print("##### file: " + fileNM)
			for read in f5.get_reads():
				raw_data = read.get_raw_data(scale=True)

				### only parse reads that are long enough
				if len(raw_data) >= (cutoff + 3000):
					if read.read_id in posli:
						pi += 1
						arrpos.append(raw_data[cutoff:(cutoff + 3000)])
						if (pi%batch == 0) and (pi != 0):
							normalization(arrpos, pi, outpath, pos = True)
							del arrpos
							arrpos = []

					if read.read_id in negli:
						ni += 1
						arrneg.append(raw_data[cutoff:(cutoff + 3000)])
						if (ni%batch == 0) and (ni != 0):
							normalization(arrneg, ni, outpath, pos = False)
							del arrneg
							arrneg = []


if __name__ == '__main__':
	main()
