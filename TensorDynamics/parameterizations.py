''' Physical parameterization functions'''


import numpy as np
import tensorflow as tf
import TensorDynamics.constants as constants

A=constants.A_EARTH #radius of earth
A2 = A*A
R=constants.DRY_AIR_CONST # gas constant
KAPPA=constants.KAPPA

def held_suarez(m_obj,state,grid_state,derivs):
	"""
	Calculate time derivative tendencies due to relaxation towards held-suarez conditions

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with spherical harmonic coefficients of each variable
		grid_state: dictionary with grid space values of each needed variable
		derivs: dictionary with spherical harmonic coefficient time tendencies of each variable
		
	Returns:
		derivs: dictionary with updated spherical harmonic coefficient time tendencies of each variable
	"""	
	# need temperature, surface pressure in grid space
	T=grid_state["T"]
	lps=grid_state["lps"]
	pressure = m_obj.sigmas[:,None,None]*tf.math.exp(lps)
	
	p0 = 100000 # reference pressure in Pa
	sig_b=0.7 # vertical limit for boundary layer drag

	# drag coefficient for horizontal winds
	kf = 1/(3600*24)
	kv = kf * tf.keras.activations.relu((m_obj.sigmas-sig_b)/(1-sig_b))[:,None,None]
	kv=tf.cast(kv,np.csingle)

	# nudging coefficient for temperature field
	ks= 1/(3600*24)/4
	ka= 1/(3600*24)/40
	kt = ka + (ks-ka)* tf.keras.activations.relu((m_obj.sigmas-sig_b)/(1-sig_b))[:,None,None] * m_obj.coslats**4

	# meridional and vertical scales of temperature and potential temperature, respectively
	delT_y=60
	delTheta_z= 10

	# calculate relaxation temperature (function of pressure and latitude)
	T_eq = (315 - delT_y * tf.math.sin(m_obj.f_obj.lats[None,:,None])**2 - delTheta_z * tf.math.log(pressure/p0)*m_obj.coslats**2)* (pressure/p0)**KAPPA
	check=tf.cast(tf.math.greater(T_eq,200),np.single) # assure that minimum relaxation temperature is 200 K
	T_eq = check*T_eq + (1-check)*200
	
	# add tendencies to existing parameterization tendencies
	derivs["chi"]=derivs["chi"]-kv*(-tf.cast(m_obj.n2,np.csingle)*state["chi_amn"])
	derivs["psi"]=derivs["psi"]-kv*(-tf.cast(m_obj.n2,np.csingle)*state["psi_amn"])
	derivs["T"]= derivs["T"]+ m_obj.f_obj.calc_sh_coeffs(-kt*(T-T_eq))
	
	return derivs


def condensation(m_obj,state,grid_state,derivs):
	"""
	Calculate time derivative tendencies due to large-scale condensation

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with spherical harmonic coefficients of each variable
		grid_state: dictionary with grid space values of each needed variable
		derivs: dictionary with spherical harmonic coefficient time tendencies of each variable
		
	Returns:
		derivs: dictionary with updated spherical harmonic coefficient time tendencies of each variable
	"""	    

	# need temperature, surface pressure, and specific humidity in grid space
	Q=grid_state["Q"]
	T=grid_state["T"]
	lps=grid_state["lps"]

	Lv=2.25e6 # latent heat of vaporization
	Rv=461. # specific gas constant of water vapor
	cp=1004. # specific heat constant
	mu_q=tf.constant(0.60779,dtype=np.single) #scaling factor from prognostic Q to specific humidity q, Q = mu_q * q

	# calculate pressures at each grid level/grid point, then get saturation vapor pressure and humidity
	pressure=(m_obj.sigmas[:,None,None]*tf.math.exp(lps))
	e_star=611.3*tf.math.exp(Lv/Rv*(1./273.15-1./T))
	q_star= (R/Rv)*e_star/pressure
	dq_star_dT = q_star*Lv/Rv/T/T
	
	# determine with grid cells are oversaturated
	oversat=tf.cast(tf.math.greater(Q,q_star*mu_q),dtype=np.single)

	# calculate tendency due to condensation, relaxation time of 6 model time steps
	nsteps=6
	dq_change = oversat*(q_star-Q/mu_q)/(1+Lv/cp*dq_star_dT)
	dQ_cond = dq_change*mu_q/(m_obj.dt*nsteps)
	dT_cond = -Lv/cp*dQ_cond/mu_q

	# convert to grid space and add to existing parameterization tendencies
	derivs["Q"]=derivs["Q"]+m_obj.f_obj.calc_sh_coeffs(dQ_cond)
	derivs["T"]=derivs["T"]+m_obj.f_obj.calc_sh_coeffs(dT_cond)
	return derivs


def convection(m_obj,state,grid_state,derivs):
	############################### CONVECTION (STILL DEBUGGING)
	'''Q=grid_state["Q"]
	T=grid_state["T"]
	lps=grid_state["lps"]
	pressure=(m_obj.sigmas[:,None,None]*tf.math.exp(lps))
	mu_q=tf.constant(0.60779,dtype=np.single)
	Lv=2.25e6 # latent heat of vaporization
	# calculate temperature for the dry adiabat based on temperature at the lowest model level
	T_dry = T[-1]*tf.math.pow(pressure/pressure[-1],KAPPA)

	nlevels=len(m_obj.sigmas)
	zeros=tf.zeros_like(T)

	T_ref=T_dry[-1][None,:,:]
	T_ref=tf.where(tf.equal(m_obj.sigmas,m_obj.sigmas[-1])[:,None,None],T_ref,zeros)

	Tv_env = T + Q # virtual temperature of the environment
	
	LZB = tf.math.exp(lps) # initialize tensor for calculating level of zero buoyancy

	# loop through each level from bottom to top, determining whether to follow dry or wet adiabat
	for k in tf.range(2,nlevels+1):
	
		# determine whether level is saturated or not
		layer_e_star=611.3*tf.math.exp(4880*(1/273.15-1/T_ref[-k+1]))
		layer_qstar = (R/461)*layer_e_star/pressure[-k+1]
		layer_sat = tf.cast(tf.math.greater(Q[-1]/mu_q,layer_qstar),dtype=np.single)
		lay_Tv = T_ref[-k+1]+mu_q*layer_qstar

		dTdZ=(1+layer_qstar*Lv/287/lay_Tv/(1-layer_qstar)**2)
		dTdZ=-dTdZ/(1+layer_qstar*Lv*Lv/461/lay_Tv/lay_Tv/1004/(1-layer_qstar)**2)

		thickness = -(tf.math.log(pressure[-k])-tf.math.log(pressure[-k+1]))*287*(lay_Tv)
		dT=thickness*dTdZ/1004
		T_wet=T_ref[-k+1]+dT

		layer_T = (1-layer_sat)*T_dry[-k]+layer_sat*T_wet

		T_ref=tf.where(tf.equal(m_obj.sigmas,m_obj.sigmas[-k])[:,None,None],layer_T,T_ref)

		is_buoyant=tf.cast(tf.greater(lay_Tv,Tv_env[-k+1]),dtype=np.single)
		LZB=is_buoyant*pressure[-k+1]+(1-is_buoyant)*LZB
	
	e_star=1000*0.6113*tf.math.exp(5423*(1./273.15-1./T_ref))
	q_ref= 0.7*(287/461)*e_star/pressure

	ZB = tf.cast(tf.math.greater(pressure,LZB),dtype=np.single)

	thicknesses=m_obj.dsigs[:,None,None]*tf.math.exp(lps)
	delT= (T_ref-T)
	delq=(q_ref-Q/mu_q)

	P_T = tf.math.reduce_sum(ZB*thicknesses*delT,axis=0)
	P_q = tf.math.reduce_sum(ZB*thicknesses*delq,axis=0)
	delP = tf.math.reduce_sum(ZB*thicknesses,axis=0)
	
	T_corrT= -P_T/delP
	T_corrq=-P_q*Lv/1004/delP
	fq= 1 + P_q/tf.math.reduce_sum(-ZB*thicknesses*q_ref,axis=0)

	isDeep=tf.cast(tf.math.less(P_q,0),dtype=np.single)
	T_ref=T_ref+T_corrT+ T_corrq * isDeep
	q_ref= (isDeep+(1-isDeep)*fq)*q_ref
	
	isConv = tf.cast(tf.math.greater(P_T,0),dtype=np.single)

	dq_conv= isConv*(q_ref*mu_q-Q)/(3600*3)
	dT_conv= isConv*(T_ref-T)/(3600*3)
	
	dq_conv=tf.where(tf.math.is_nan(dq_conv),zeros,dq_conv)
	dT_conv=tf.where(tf.math.is_nan(dT_conv),zeros,dT_conv)

	derivs["Q"]=derivs["Q"]+m_obj.f_obj.calc_sh_coeffs(dq_conv)
	derivs["T"]=derivs["T"]+m_obj.f_obj.calc_sh_coeffs(dT_conv)'''
	return derivs

# create dictionary holding each parameterization method
all_params={}
all_params["convection"]=convection
all_params["condensation"]=condensation
all_params["held_suarez"]=held_suarez

# create dictionary holding names of needed variables in grid space for each parameterization
grid_vars={}
grid_vars["convection"]=["Q","T","lps"]
grid_vars["condensation"]=["Q","T","lps"]
grid_vars["held_suarez"]=["T","lps"]