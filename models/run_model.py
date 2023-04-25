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
	weights_path = "models/weights/resnet_1024/model_30012023_183630"
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
	model.eval()
	model.to('cpu')

	# Normalize
	sequence = z_normalization(sequence, decimals=5)

	# Split timeseries and load to cpu
	sequence = split_ts(sequence, window_size)[:, np.newaxis]
	sequence = torch.from_numpy(sequence).to('cpu')

	# Generate predictions
	preds = model(sequence.float()).argmax(dim=1).tolist()

	# Majority voting
	counter = Counter(preds)
	most_voted = counter.most_common(1)
	detector = most_voted[0][0]
	
	counter = dict(counter)
	vote_summary = {detector_names[key]:counter[key] for key in counter}
	
	return detector_names[detector], vote_summary
