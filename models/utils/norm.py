########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Spring Semester 2023
# @where : LIPADE / FORTH
# @title : ADecimo
# @component: models/utils
# @file : norm
#
########################################################################

import numpy as np

def z_normalization(ts, decimals=5):
	# Z-normalization (all windows with the same value go to 0)
	if len(set(ts)) == 1:
		ts = ts - np.mean(ts)
	else:
		ts = (ts - np.mean(ts)) / np.std(ts)
	ts = np.around(ts, decimals=decimals)

	# Test normalization
	assert(
		np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
	), "After normalization it should: mean == 0 and std == 1"

	return ts