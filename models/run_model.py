########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Spring Semester 2023
# @where : LIPADE / FORTH
# @title : ADecimo
# @component: models
# @file : run_model
#
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch

from models.model.resnet import ResNetBaseline
from models.utils.split_ts import split_ts
from models.utils.norm import z_normalization

# Detector (NOTE: run_model.py requires THIS specific order)
detector_names = [
	'AE', 
	'CNN', 
	'HBOS', 
	'IFOREST', 
	'IFOREST1', 
	'LOF', 
	'LSTM', 
	'MP', 
	'NORMA', 
	'OCSVM', 
	'PCA', 
	'POLY'
]

def run_model(sequence):
	"""
	"""
	weights_path = "models/weights/resnet_1024/model_10042023_172539"
	window_size = 1024

	# Load model
	model = ResNetBaseline(
		in_channels=1, 
		mid_channels=64,
		num_pred_classes=12,
		num_layers=3
	)

	# Load weights
	model.load_state_dict(torch.load(weights_path, map_location='cpu'))
	model.to('cpu')

	# Normalize
	sequence = z_normalization(sequence, decimals=5)

	# Split timeseries
	sequence = split_ts(sequence, window_size)[:, np.newaxis]
	plt.plot(sequence[0, 0])
	plt.show()

	sequence = torch.from_numpy(sequence).to('cpu')

	# Generate predictions
	preds = model(sequence.float()).argmax(dim=1).tolist()

	# Majority voting
	counter = Counter(preds)
	most_voted = counter.most_common(1)
	detector = most_voted[0][0]

	return detector_names[detector], str(sequence.shape)


'''
if __name__ == "__main__":
	
	# Signal set up
	Fs = 80
	f = 5
	samples = 8000

	# Anomalies set up
	noise_power = 0.8
	n_anomalies = 1000
	mean_anomalies_len = 50
	anomalies_len_std = 10

	# Generate signal
	x = np.arange(samples)
	y = np.sin(2 * np.pi * f * x / Fs) # + noise 

	# Generate noise
	anomalies_pos = np.random.randint(0, samples, n_anomalies)
	anomalies_len = np.ceil(np.random.normal(mean_anomalies_len, anomalies_len_std, n_anomalies))
	noise = noise_power * np.random.normal(0, 1, samples)

	# Create anomalies mask
	noise_mask = np.zeros(samples)
	for pos, length in zip(anomalies_pos, anomalies_len):
		start = int(max(pos - length // 2, 0))
		end = int(min(pos + length // 2, samples))
		
		noise_mask[start:end] = 1
	noise *= noise_mask
	
	# Combine
	sequence = y + noise

	# plt.plot(x, sequence)
	# plt.show()

	pred_detector = run_model(sequence)

	print(pred_detector)
'''