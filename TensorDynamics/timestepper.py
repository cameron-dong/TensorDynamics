import numpy as np
import tensorflow as tf

from TensorDynamics.time_integration import euler, leap, rate_of_change, physics, euler_BE, explicit_step, add_diffusion_tend, perform_diffusion, linear

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
	dchi, dpsi, dT, dlps, dq = rate_of_change(m_obj,state)

	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)		
		dq=dq+dq_phys
		dT=dT+dT_phys

	# placeholder values
	tmp_old={}
	tmp_old["chi_amn"]=tf.identity(state["chi_amn"])
	tmp_old["psi_amn"]=tf.identity(state["psi_amn"])
	tmp_old["T_amn"]=tf.identity(state["T_amn"])
	tmp_old["lps_amn"]=tf.identity(state["lps_amn"])
	tmp_old["Q_amn"]=tf.identity(state["Q_amn"])
	tmp_old["Zs_amn"]=tf.identity(state["Zs_amn"])
	
	# do semi-implicit leapfrog step
	state=leap(m_obj,state,old,dT,dlps,dchi,dpsi,dq,tf.cast(dt,np.csingle))
	state=perform_diffusion(m_obj,state,tf.cast(dt,np.csingle))
	# asselin-roberts-williams filtering
	sig=0.53
	eta=0.03
	out_old={}
	f=(old["psi_amn"]-2*tmp_old["psi_amn"]+state["psi_amn"])
	out_old["psi_amn"]=(tmp_old["psi_amn"]+sig*eta*f)
	state["psi_amn"]=state["psi_amn"]-(1-sig)*eta*f

	f=(old["T_amn"]-2*tmp_old["T_amn"]+state["T_amn"])
	out_old["T_amn"]=(tmp_old["T_amn"]+sig*eta*f)
	state["T_amn"]=state["T_amn"]-(1-sig)*eta*f
	
	f=(old["chi_amn"]-2*tmp_old["chi_amn"]+state["chi_amn"])
	out_old["chi_amn"]=(tmp_old["chi_amn"]+sig*eta*f)
	state["chi_amn"]=state["chi_amn"]-(1-sig)*eta*f

	f=(old["lps_amn"]-2*tmp_old["lps_amn"]+state["lps_amn"])
	out_old["lps_amn"]=(tmp_old["lps_amn"]+sig*eta*f)
	state["lps_amn"]=state["lps_amn"]-(1-sig)*eta*f

	f=(old["Q_amn"]-2*tmp_old["Q_amn"]+state["Q_amn"])
	out_old["Q_amn"]=(tmp_old["Q_amn"]+sig*eta*f)
	state["Q_amn"]=state["Q_amn"]-(1-sig)*eta*f

	out_old["Zs_amn"]=old["Zs_amn"]
	return state, out_old

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
	dchi, dpsi, dT, dlps, dq = rate_of_change(m_obj,state)
	
	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)
		dq=dq+dq_phys
		dT=dT+dT_phys
	
	# do semi-implicit forward euler step
	state=euler(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt,np.csingle),m_obj.imp_inv_eul)
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

	# placeholder dictionary to hold onto current state
	start_state={}
	start_state["chi_amn"]=tf.identity(state["chi_amn"])
	start_state["psi_amn"]=tf.identity(state["psi_amn"])
	start_state["T_amn"]=tf.identity(state["T_amn"])
	start_state["lps_amn"]=tf.identity(state["lps_amn"])
	start_state["Q_amn"]=tf.identity(state["Q_amn"])
	start_state["Zs_amn"]=tf.identity(state["Zs_amn"])

	####################### FIRST CYCLE
	# calculate explicit terms
	dchi, dpsi, dT, dlps, dq = rate_of_change(m_obj,state)
	dchi, dpsi, dT, dlps, dq = add_diffusion_tend(m_obj,state,dchi, dpsi, dT, dlps, dq)
	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)
		dq=dq+dq_phys
		dT=dT+dT_phys

	# instead of trapezoidal, do a modified backward euler
	newstate=euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	# calculate linear terms, then perform a fully explicit step
	L_chi, L_psi, L_T, L_lps,L_q = linear(m_obj,state)
	L_chi2, L_psi2, L_T2, L_lps2,L_q2 = linear(m_obj,newstate)
	L_chi = (L_chi+L_chi2)/2
	L_T = (L_T+L_T2)/2
	L_lps = (L_lps+L_lps2)/2

	state=explicit_step(m_obj,state,dT+L_T,dlps+L_lps,dchi+L_chi,dpsi+L_psi,dq+L_q,tf.cast(dt/3,np.csingle))

	#################################### SECOND CYCLE
	dchi2, dpsi2, dT2, dlps2, dq2 = rate_of_change(m_obj,state)
	dchi2, dpsi2, dT2, dlps2, dq2= add_diffusion_tend(m_obj,state,dchi2, dpsi2, dT2, dlps2, dq2)

	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)
		dq2=dq2+dq_phys
		dT2=dT2+dT_phys
	w=weights[1]

	L_chi, L_psi, L_T, L_lps,L_q = linear(m_obj,start_state)
	L_chi2, L_psi2, L_T2, L_lps2,L_q2 = linear(m_obj,state)
	L_chi = (L_chi-L_chi2)/2
	L_T = (L_T-L_T2)/2
	L_lps = (L_lps-L_lps2)/2

	dchi = ((1-w)*dchi+w*dchi2)#+L_chi
	dpsi = ((1-w)*dpsi+w*dpsi2)
	dT = ((1-w)*dT+w*dT2)#+L_T
	dlps = ((1-w)*dlps+w*dlps2)#+L_lps
	dq = ((1-w)*dq+w*dq2)

	state=euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	########################################### THIRD CYCLE
	dchi3, dpsi3, dT3, dlps3, dq3 = rate_of_change(m_obj,state)
	dchi3, dpsi3, dT3, dlps3, dq3= add_diffusion_tend(m_obj,state,dchi3, dpsi3, dT3, dlps3, dq3)

	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)
		dq3=dq3+dq_phys
		dT3=dT3+dT_phys
	w=weights[2]

	dchi = ((1-w)*dchi+w*dchi3)
	dpsi = ((1-w)*dpsi+w*dpsi3)
	dT = ((1-w)*dT+w*dT3)
	dlps = ((1-w)*dlps+w*dlps3)
	dq = ((1-w)*dq+w*dq3)

	newstate=euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)

	L_chi, L_psi, L_T, L_lps,L_q = linear(m_obj,start_state)
	L_chi2, L_psi2, L_T2, L_lps2,L_q2 = linear(m_obj,state)
	L_chi3, L_psi3, L_T3, L_lps3,L_q3 = linear(m_obj,newstate)

	L_chi = ((L_chi+L_chi2)/2-L_chi3)/4
	L_T = ((L_T+L_T2)/2-L_T3)/4
	L_lps = ((L_lps+L_lps2)/2-L_lps3)/4

	c=1
	dchi=c*dchi+L_chi
	dpsi=c*dpsi+L_psi
	dT=c*dT+L_T
	dlps=c*dlps+L_lps
	dq=c*dq+L_q

	state=euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt/3,np.csingle),m_obj.imp_inv_SIL3)
	
	return state

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
	dchi, dpsi, dT, dlps, dq = rate_of_change(m_obj,state)
	if m_obj.do_physics:
		dq_phys, dT_phys = physics(m_obj,state)
		dq=dq+dq_phys
		dT=dT+dT_phys

	# perform initial step with euler forward for explicit terms and euler backward for implicit
	newstate=euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,tf.cast(dt,np.csingle),m_obj.imp_inv)

	# use initial step to update the rate of change for the nonlinear terms
	dchi2, dpsi2, dT2, dlps2, dq2 = rate_of_change(m_obj,newstate)
	if m_obj.do_physics:
		dq2=dq2+dq_phys
		dT2=dT2+dT_phys

	dchi= (dchi+dchi2)/2
	dpsi= (dpsi+dpsi2)/2
	dT= (dT+dT2)/2
	dlps= (dlps+dlps2)/2
	dq= (dq+dq2)/2

	# calculated averaged rate of change for the linear terms
	L_chi, L_psi, L_T, L_lps,L_q = linear(m_obj,state)
	L_chi2, L_psi2, L_T2, L_lps2,L_q2 = linear(m_obj,newstate)
	L_chi = (L_chi+L_chi2)/2
	L_T = (L_T+L_T2)/2
	L_lps = (L_lps+L_lps2)/2

	# do a fully explicit step using new rate of change
	state=explicit_step(m_obj,state,dT+L_T,dlps+L_lps,dchi+L_chi,dpsi+L_psi,dq+L_q,tf.cast(dt,np.csingle))

	return state

