import numpy as np
import tensorflow as tf

from TensorDynamics.time_integration import euler, leap, rate_of_change


##########################################################################################################################

@tf.function(jit_compile=True)
def leap_step(m_obj,dt,state,old):
	"""
	Performs one leapfrog step forward in time

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.Variable's with current spherical harmonic coefficients
		old (dict): tf.Variable's with spherical harmonic coefficients from previous step

	Returns:
		coefficient tf.Variable's in 'state' and 'old' are updated in place.
	"""
	
	# calculate explicit terms
	dchi, dpsi, dT, dlps = rate_of_change(m_obj,state)

	# placeholder values
	tmp_old=[state["psi_amn"]*1,state["T_amn"]*1,state["chi_amn"]*1,state["lps_amn"]*1]
	
	# do semi-implicit leapfrog step
	leap(m_obj,state,old,dT,dlps,dchi,dpsi,tf.cast(dt,np.csingle))
	
	# asselin-roberts filtering
	old["psi_amn"].assign(tmp_old[0]+0.03*(old["psi_amn"]-2*tmp_old[0]+state["psi_amn"]))
	old["T_amn"].assign(tmp_old[1]+0.03*(old["T_amn"]-2*tmp_old[1]+state["T_amn"]))
	old["chi_amn"].assign(tmp_old[2]+0.03*(old["chi_amn"]-2*tmp_old[2]+state["chi_amn"]))
	old["lps_amn"].assign(tmp_old[3]+0.03*(old["lps_amn"]-2*tmp_old[3]+state["lps_amn"]))
	return None

#######################################################################################################

@tf.function(jit_compile=True)
def euler_step(m_obj,dt,state):
	"""
	Performs one euler step forward in time

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.Variable's with current spherical harmonic coefficients

	Returns:
		coefficient tf.Variable's in 'state' are updated in place.
	"""
	
	# calculate explicit terms
	dchi, dpsi, dT, dlps = rate_of_change(m_obj,state)
	
	# do semi-implicit forward euler step
	euler(m_obj,state,dT,dlps,dchi,dpsi,tf.cast(dt,np.csingle))
	return None


#######################################################################################################
'''
Semi-implicit Lorenz N-Cycle
Not implemented

Hotta, D., Kalnay, E. and Ullrich, P., 2016. A Semi-Implicit Modification to the Lorenz N-Cycle Scheme
and Its Application for Integration of Meteorological Equations. Monthly Weather Review, 144(6), pp.2215-2233.


@tf.function(jit_compile=True)
def SIL4(m_obj,dt,state,k,dchi,dpsi,dT,dlps, N=4):

	w=tf.cast(N/(N-tf.math.mod(k,N)),np.csingle)
	dchi2, dpsi2, dT2, dlps2 = rate_of_change(m_obj,state)

	dchi= w*dchi2+(1-w)*dchi
	dpsi= w*dpsi2+(1-w)*dpsi
	dT = w*dT2+(1-w)*dT
	dlps= w*dlps2+(1-w)*dlps

	euler(m_obj,state,dT,dlps,dchi,dpsi,tf.cast(dt,np.csingle))

	return dchi, dpsi, dT, dlps
'''
