########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Spring Semester 2023
# @where : LIPADE / FORTH
# @title : ADecimo
# @component: models/utils
# @file : split_ts
#
########################################################################


import math
import numpy as np

def split_ts(data, window_size):
	'''Split a timeserie into windows according to window_size.
	If the timeserie can not be divided exactly by the window_size
	then the first window will overlap the second.

	:param data: the timeserie to be segmented
	:param window_size: the size of the windows
	:return data_split: an 2D array of the segmented time series
	'''

	# Compute the modulo
	modulo = data.shape[0] % window_size

	# Compute the number of windows
	k = data[modulo:].shape[0] / window_size
	assert(math.ceil(k) == k)

	# Split the timeserie
	data_split = np.split(data[modulo:], k)
	if modulo != 0:
		data_split.insert(0, list(data[:window_size]))
	data_split = np.asarray(data_split)

	return data_split