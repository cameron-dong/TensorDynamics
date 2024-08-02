import numpy as np
import tensorflow as tf
import TensorDynamics.constants as constants

A=constants.A_EARTH #radius of earth
A2=A*A
R=constants.DRY_AIR_CONST # gas constant
OMEGA=constants.OMEGA
KAPPA=constants.KAPPA

# placeholder for tf.constant
def tfc(a,dtype=np.single):
	return tf.constant(a,dtype=dtype)

# placeholder for tf.Variable
def tfv(a,dtype=np.single):
	return tf.Variable(a,dtype=dtype)

# Useful function for calculating matrices needed for semi-implicit method
def L(x):
	if x>=0:
		return 1
	else:
		return 0
	
def vdiff_inds(nlevels):
		"""
		Creates useful index arrays for calculating vertical finite differences, given number of vertical levels
		"""
		inds=np.array([i for i in range(nlevels)])

		# positive inds
		pinds=(inds*1)
		pinds[-1]=(-1)
		pinds=(pinds+1)

		# negative inds
		ninds=(inds*1)
		ninds[0]=(nlevels)
		ninds=(ninds-1)
		return inds, pinds, ninds


def get_matrices(T_bar,sigmas,alphas,dsigs,dt,n2):
	"""
	Create matrices that are needed for the leapfrog-trapezoidal time step
	Args:
		T_bar: reference vertical temperature profile
		sigmas: model sigma levels
		alphas: model alpha values
		dsigs: model level thicknesses
		dt: model timestep size
		n2: square of spherical harmonic total wavenumber matrix
	Returns:
		G_mat: tensor with dimensions (levels, levels)
		H_mat: tensor with dimensions (levels, levels)
		imp_inv: tensor with dimensions (levels, levels, total_wavenumber, zonal wavenumber)
	"""

	# initialize matrices with zeros
	G_mat=tf.Variable(tf.zeros((len(sigmas),len(sigmas))))
	H_mat=tf.Variable(tf.zeros(np.shape(G_mat),dtype=np.single))


	# loop through to assign values
	for i in range(len(alphas)):
		for j in range(len(alphas)):
			if i==j:
				G_mat[i,j].assign(alphas[j])
			elif j>i:
				G_mat[i,j].assign(alphas[j]+alphas[j-1])
			H_mat[i,j].assign(KAPPA*T_bar[i]*L(i-j)/dsigs[i]*(alphas[i]+alphas[i-1]*L(i-j-1)))
			if i<(len(alphas)-1):
				H_mat[i,j].assign(H_mat[i,j]-(T_bar[i+1]-T_bar[i])/(dsigs[i+1]+dsigs[i])*(L(i-j)-np.sum(dsigs[0:i+1])))
			H_mat[i,j].assign(H_mat[i,j]*dsigs[j])
	G_mat.assign(G_mat*R)

	# convert to tf.tensor
	G_mat=tfc(G_mat)
	H_mat=tfc(H_mat)

	# Create inversion matrix for semi-implicit solution of divergence
	B_mat = (G_mat@H_mat+R*T_bar[:,None]@dsigs[None,:])
	imp=(tf.eye(len(alphas))[:,:,None,None]+dt*dt*B_mat[:,:,None,None]*n2[None,None,:,:]/A2)
	imp_inv=tf.constant(np.moveaxis(np.linalg.inv(np.moveaxis(imp,(0,1),(2,3))),(2,3),(0,1)))
	
	return G_mat, H_mat, imp_inv

def get_lorenz_matrices(T_bar,sigmas,alphas,dsigs,dt,n2):
	"""
	Create matrices that are needed for the euler-trapezoidal time step
	Args:
		T_bar: reference vertical temperature profile
		sigmas: model sigma levels
		alphas: model alpha values
		dsigs: model level thicknesses
		dt: model timestep size
		n2: square of spherical harmonic total wavenumber matrix
	Returns:
		G_mat: tensor with dimensions (levels, levels)
		H_mat: tensor with dimensions (levels, levels)
		imp_inv_eul: tensor with dimensions (levels, levels, total_wavenumber, zonal wavenumber)
	"""
	# initialize matrices with zeros
	G_mat=tf.Variable(tf.zeros((len(sigmas),len(sigmas))))
	H_mat=tf.Variable(tf.zeros(np.shape(G_mat),dtype=np.single))

	for i in range(len(alphas)):
		for j in range(len(alphas)):
			if i==j:
				G_mat[i,j].assign(alphas[j])
			elif j>i:
				G_mat[i,j].assign(alphas[j]+alphas[j-1])
			H_mat[i,j].assign(KAPPA*T_bar[i]*L(i-j)/dsigs[i]*(alphas[i]+alphas[i-1]*L(i-j-1)))
			if i<(len(alphas)-1):
				H_mat[i,j].assign(H_mat[i,j]-(T_bar[i+1]-T_bar[i])/(dsigs[i+1]+dsigs[i])*(L(i-j)-np.sum(dsigs[0:i+1])))
			H_mat[i,j].assign(H_mat[i,j]*dsigs[j])
	G_mat.assign(G_mat*R)
	# convert to tf.tensor
	G_mat=tfc(G_mat)
	H_mat=tfc(H_mat)

	# Create inversion matrix for semi-implicit solution of divergence
	B_mat = (G_mat@H_mat+R*T_bar[:,None]@dsigs[None,:])
	imp=(tf.eye(len(alphas))[:,:,None,None]+(dt*dt/4.)*B_mat[:,:,None,None]*n2[None,None,:,:]/A2)
	imp_inv=tf.constant(np.moveaxis(np.linalg.inv(np.moveaxis(imp,(0,1),(2,3))),(2,3),(0,1)))
	return G_mat, H_mat, imp_inv


def get_levels(nlevels):
	"""
	Given number of levels, create equidistant sigma level grid
	Currently only accepts nlevels=2, 5, 10, 20, 25, or 40

	Args: 
		nlevels: number of sigma levels, 
	Returns:
		sigmas: sigma levels
		alphas: sigma level alpha values
		dsigs: sigma level thicknesses
		sigs_i: sigma level interfaces
	"""
	if nlevels==5:
		delta_sig=0.2
		sigmas=([0.1, 0.3, 0.5, 0.7, 0.9])
	elif nlevels==10:
		delta_sig=0.1
		sigmas=([0.05+0.1*i for i in range(10)])
	elif nlevels==20:
		delta_sig=0.05
		sigmas=([0.025+0.05*i for i in range(20)])
	elif nlevels==25:
		delta_sig=0.04
		sigmas=([0.02+0.04*i for i in range(25)])
	elif nlevels==40:
		delta_sig=0.025
		sigmas=([0.0125+0.025*i for i in range(40)])
	elif nlevels==2:
		delta_sig=0.5
		sigmas=[0.25, 0.75]

	sigmas=tfc(sigmas)
	dsigs=tfc([delta_sig for i in range(nlevels)])
	sigs_i=tfc([delta_sig*i for i in range(nlevels+1)])

	alphas=tfv(sigmas)
	alphas[-1].assign(-np.log(sigmas[-1]))
	alphas[0:-1].assign(0.5*np.log(sigmas[1:]/sigmas[:-1]))
	alphas=tfc(alphas)

	return sigmas, alphas, dsigs, sigs_i