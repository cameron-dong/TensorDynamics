import tensorflow as tf
import numpy as np

def pressure_to_sigma(PS,pressures,sigmas, ys):
	"""
	Vertical linear interpolation from pressure to sigma coordinates

	Adapted from https://brentspell.com/blog/2022/tensorflow-interp/ to include batch dimensions

	Args:
		PS: tensor with dimensions (latitude,longitude) containing surface pressure in Pa
		pressures: tensor with dims (level) containg pressure in hPa
		sigmas: tensor with dims (level) containing target sigmas
		ys: tensor with dims (level, latitude, longitude) containing values at input pressure levels

	Returns:
		y_targ: tensor with dims (level, latitude, longitude) containing values at target sigma levels

	"""

	# create 3-dimensional arrays with original and target pressures
	xs=pressures[:,None,None]+PS*0
	x_targ=sigmas[:,None,None]*PS/100
	
	# convert to float64 for the interpolation
	ys=tf.constant(ys)
	ys = tf.cast(ys, tf.float64)
	xs = tf.cast(xs, tf.float64)
	x_targ = tf.cast(x_targ, tf.float64)

	# pad control points for extrapolation
	xs = tf.concat([[xs.dtype.min*tf.ones(ys.shape[1:],dtype=tf.float64)], xs, [xs.dtype.max*tf.ones(ys.shape[1:],dtype=tf.float64)]], axis=0)
	ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

	# compute slopes, pad at the edges to flatten
	ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
	ms = tf.pad(ms[:-1], [(1, 1),(0,0),(0,0)])
	# solve for intercepts
	bs = ys - ms*xs
	
	# search for the line parameters at each input data point
	# create a grid of the inputs and piece breakpoints for thresholding
	# rely on argmax stopping on the first true when there are duplicates,
	i = tf.math.argmax(xs[tf.newaxis, ...] > x_targ[:, tf.newaxis,:,:], axis=1)

	# transpose so that latitude, longitude can be used as batch dimensions
	m=tf.gather(tf.transpose(ms,[1,2,0]),tf.transpose(i,[1,2,0]),batch_dims=2)
	b=tf.gather(tf.transpose(bs,[1,2,0]),tf.transpose(i,[1,2,0]),batch_dims=2)

	# transpose back to original dimension ordering
	m=tf.transpose(m,[2,0,1])
	b=tf.transpose(b,[2,0,1])

	# apply the linear mapping at each input data point
	y_targ = tf.cast(m*x_targ + b, np.single)
	
	return y_targ