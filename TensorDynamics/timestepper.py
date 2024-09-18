''' wrappers for time integration methods, which calculate time tendencies then perform appropriate time step'''

import numpy as np
import tensorflow as tf

from TensorDynamics.time_integration import euler, leap, euler_BE, explicit_step,  perform_diffusion
from TensorDynamics.get_tendencies  import rate_of_change, physics, add_diffusion_tend, linear
##########################################################################################################################
do_jit=True

@tf.function(jit_compile=do_jit)
def leap_step(m_obj,dt,state,old):
	"""
	Performs one leapfrog step forward in time

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.tensor's with current spherical harmonic coefficients
		old (dict): tf.tensor's with spherical harmonic coefficients from previous step

	Returns:
		state (dict): tf.tensor's with updated spherical harmonic coefficients
		out_old (dict): tf.tensor's with filtered spherical harmonic coefficients from current step
	"""
	
	# calculate explicit terms
	explicit_derivs = rate_of_change(m_obj,state)

	# add physics tendencies
	if m_obj.do_physics:
		p_derivs = physics(m_obj,old)		
		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs[vari]=explicit_derivs[vari]+p_derivs[vari[:-4]]

	# placeholder values
	tmp_old={}
	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		tmp_old[vari]=tf.identity(state[vari])
	
	# do semi-implicit leapfrog step
	state=leap(m_obj,state,old,explicit_derivs,tf.cast(dt,np.csingle))
	state=perform_diffusion(m_obj,state,tf.cast(dt,np.csingle))

	# asselin-roberts-williams filtering
	sig=0.53
	eta=0.03
	out_old={}
	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		f=(old[vari]-2*tmp_old[vari]+state[vari])
		out_old[vari]=(tmp_old[vari]+sig*eta*f)
		state[vari]=state[vari]-(1-sig)*eta*f
	out_old["Zs_amn"]=old["Zs_amn"]

	return state, out_old

#############################

@tf.function(jit_compile=do_jit)
def heuns_step(m_obj,dt,state):
	"""
	Performs modified euler step

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.tensor's with current spherical harmonic coefficients

	Returns:
		state (dict): tf.tensor's with updated spherical harmonic coefficients
	"""
	# calculate explicit terms
	explicit_derivs = rate_of_change(m_obj,state)
	if m_obj.do_physics:
		p_derivs = physics(m_obj,state)		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs[vari]=explicit_derivs[vari]+p_derivs[vari[:-4]]

	# perform initial step with euler forward for explicit terms and euler backward for implicit
	newstate=euler_BE(m_obj,state,explicit_derivs,tf.cast(dt,np.csingle),m_obj.imp_inv)

	# calculated averaged rate of change for the linear terms
	l_derivs = linear(m_obj,state)
	l_derivs_2 = linear(m_obj,newstate)

	# use initial step to update the rate of change for the nonlinear terms
	explicit_derivs2 = rate_of_change(m_obj,newstate)
	if m_obj.do_physics:
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs2[vari]=(explicit_derivs2[vari]+p_derivs[vari[:-4]])

	# average derivative values from previous step and prediction
	total_derivs={}
	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			total_derivs[vari]=(explicit_derivs2[vari]+explicit_derivs[vari]+l_derivs_2[vari]+l_derivs[vari])/2


	# do a fully explicit step using new rate of change
	state=explicit_step(m_obj,state,total_derivs,tf.cast(dt,np.csingle))

	return state

#######################################################################################################

@tf.function(jit_compile=do_jit) # NOT USED ANYMORE
def euler_step(m_obj,dt,state):
	"""
	Performs one euler step forward in time

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.Variable's with current spherical harmonic coefficients

	Returns:
		state (dict): tf.tensor's with updated spherical harmonic coefficients
	"""
	
	# calculate explicit terms
	explicit_derivs = rate_of_change(m_obj,state)
	
	if m_obj.do_physics:
		p_derivs = physics(m_obj,state)		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs[vari]=explicit_derivs[vari]+p_derivs[vari[:-4]]
	
	# do semi-implicit forward euler step
	state=euler(m_obj,state,explicit_derivs,tf.cast(dt,np.csingle),m_obj.imp_inv_eul)
	return state
##################################################################################################

'''
Semi-implicit Lorenz N-Cycle

Whitaker, J.S. and Kar, S.K., 2013. Implicit–explicit Runge–Kutta methods for fast–slow wave problems. Monthly weather review, 141(10), pp.3426-3434.

Modified version of Butcher Tableau from above reference. Try to perform it using only one-presaved inverted matrix using a series of backward
Euler steps
'''

@tf.function(jit_compile=do_jit)
def SIL3(m_obj,dt,state):
	"""
	Performs Lorenz 3-cycle

	Args:
		m_obj: model object
		dt (float): value of timestep in seconds
		state (dict): tf.tensor's with current spherical harmonic coefficients
	Returns:
		state (dict): tf.tensor's with updated spherical harmonic coefficients
	"""
	# weights for 3-cycle
	weights=[1,3/2,3]

	####################### FIRST CYCLE
	# calculate explicit terms
	explicit_derivs = rate_of_change(m_obj,state)
	explicit_derivs = add_diffusion_tend(m_obj,state,explicit_derivs)
	if m_obj.do_physics:
		p_derivs = physics(m_obj,state)		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs[vari]=explicit_derivs[vari]+p_derivs[vari[:-4]]

	# instead of trapezoidal, do a modified backward euler
	newstate=euler_BE(m_obj,state,explicit_derivs,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	# calculate linear terms, then perform a fully explicit step
	lin_derivs = linear(m_obj,state)
	lin_derivs2 = linear(m_obj,newstate)

	total_derivs={}
	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		total_derivs[vari]= explicit_derivs[vari]+(lin_derivs[vari]+lin_derivs2[vari])/2

	newstate=explicit_step(m_obj,state,total_derivs,tf.cast(dt/3,np.csingle))

	#################################### SECOND CYCLE
	explicit_derivs2 = rate_of_change(m_obj,state)
	explicit_derivs2 = add_diffusion_tend(m_obj,state,explicit_derivs2)

	if m_obj.do_physics:
		p_derivs = physics(m_obj,newstate)		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs2[vari]=explicit_derivs2[vari]+p_derivs[vari[:-4]]
	w=weights[1]

	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		explicit_derivs[vari]= (1-w)*explicit_derivs[vari]+w*explicit_derivs2[vari] + (lin_derivs[vari]-lin_derivs2[vari])/2

	newstate=euler_BE(m_obj,newstate,explicit_derivs,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	########################################### THIRD CYCLE
	explicit_derivs3 = rate_of_change(m_obj,newstate)
	explicit_derivs3 = add_diffusion_tend(m_obj,newstate,explicit_derivs3)

	if m_obj.do_physics:
		p_derivs = physics(m_obj,newstate)		
		for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
			explicit_derivs3[vari]=explicit_derivs3[vari]+p_derivs[vari[:-4]]
	w=weights[2]

	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		explicit_derivs[vari]= (1-w)*explicit_derivs[vari]+w*explicit_derivs3[vari]

	tmp_state=euler_BE(m_obj,newstate,explicit_derivs,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	lin_derivs_3 = linear(m_obj,newstate)
	lin_derivs_4 = linear(m_obj,tmp_state)

	for vari in ["psi_amn","T_amn","chi_amn","lps_amn","Q_amn"]:
		explicit_derivs[vari]= explicit_derivs[vari]+((lin_derivs[vari]+lin_derivs_3[vari])/2-lin_derivs_4[vari])/4


	newstate=euler_BE(m_obj,newstate,explicit_derivs,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)
	
	return newstate

