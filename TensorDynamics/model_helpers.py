import numpy as np
import tensorflow as tf
import TensorDynamics.constants as constants

A=constants.A_EARTH #radius of earth
A2=A*A
R=constants.DRY_AIR_CONST # gas constant
OMEGA=constants.OMEGA
KAPPA=constants.KAPPA
GRAV=constants.GRAVITY_ACC

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
	G_mat=(np.zeros((len(sigmas),len(sigmas))))
	H_mat=(np.zeros(np.shape(G_mat),dtype=np.single))

	# loop through to assign values
	for i in range(len(alphas)):
		for j in range(len(alphas)):
			if i==j:
				G_mat[i,j]=alphas[j]
			elif j>i:
				G_mat[i,j]=alphas[j]+alphas[j-1]
			H_mat[i,j]=(KAPPA*T_bar[i]*L(i-j)/dsigs[i]*(alphas[i]+alphas[i-1]*L(i-j-1)))
			if i<(len(alphas)-1):
				H_mat[i,j]=(H_mat[i,j]-(T_bar[i+1]-T_bar[i])/(dsigs[i+1]+dsigs[i])*(L(i-j)-np.sum(dsigs[0:i+1])))
			H_mat[i,j]=H_mat[i,j]*dsigs[j]
	G_mat=G_mat*R

	# convert to tf.tensor
	G_mat=tf.constant(G_mat,dtype=np.single)
	H_mat=tf.constant(H_mat,dtype=np.single)

	# Create inversion matrix for semi-implicit solution of divergence
	B_mat = G_mat@H_mat + R*T_bar[:,None]@dsigs[None,:]
	imp=tf.eye(len(alphas))[:,:,None,None] + dt*dt*B_mat[:,:,None,None]*n2[None,None,:,:]/A2
	imp_inv=tf.constant(np.moveaxis(np.linalg.inv(np.moveaxis(imp,(0,1),(2,3))),(2,3),(0,1)))
	
	return G_mat, H_mat, imp_inv



def get_levels(nlevels):
	"""
	Given number of levels, create equidistant sigma level grid

	Args: 
		nlevels: number of sigma levels, 
	Returns:
		sigmas: sigma levels
		alphas: sigma level alpha values
		dsigs: sigma level thicknesses
		sigs_i: sigma level interfaces
		T_bar: reference temperature profile values
	"""

	delta_sig=1./nlevels

	sigmas=tf.constant([(0.5+i)*delta_sig for i in range(nlevels)],dtype=np.single)
	dsigs=tf.constant([delta_sig for i in range(nlevels)],dtype=np.single)
	sigs_i=tf.constant([delta_sig*i for i in range(nlevels+1)],dtype=np.single)

	alphas=np.array(sigmas)
	alphas[-1]=(-np.log(sigmas[-1]))
	alphas[0:-1]=(0.5*np.log(sigmas[1:]/sigmas[:-1]))
	alphas=tf.constant(alphas,dtype=np.single)
	
	T_bar=tf.constant(sigmas*0+300,dtype=np.single)

	return sigmas, alphas, dsigs, sigs_i, T_bar