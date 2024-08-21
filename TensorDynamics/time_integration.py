import numpy as np
import tensorflow as tf
import TensorDynamics.sphere_harm_new as sh
import TensorDynamics.constants as constants
from TensorDynamics.integration_helpers import to_grid_space, get_Gk, vertical_sums, do_vdiffs, do_ffts, do_gauss_quad, grid_space_transform, mat_mul_4, mat_mul_2
import matplotlib.pyplot as plt

A=constants.A_EARTH #radius of earth
A2 = A*A
R=constants.DRY_AIR_CONST # gas constant
KAPPA=constants.KAPPA


def rate_of_change(m_obj,state):
	'''
	Calculates the explicit portions of the time derivative for each variable

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with current spherical harmonic coefficients of each variable
	Returns:
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
	'''

	# from spherical harmonics, calculate values that we need in grid space
	vort, div, T_prime, dlps_dmu, dlps_dlam, U_data, V_data, Q=to_grid_space(m_obj,state)

	# treatment of vertical differences and multiplication of nonlinear terms
	Gk_adv,Gk = get_Gk(m_obj.f_obj,div,dlps_dlam,U_data,dlps_dmu,V_data)
	
	H_gs, sig_tot,sig_adv,trip_adv,trip_tot= vertical_sums(m_obj,Gk,Gk_adv,m_obj.dsigs,m_obj.alphas,m_obj.sigmas)
	
	V_vert, U_vert, T_prime_vert,Tbar_vert ,Q_vert=do_vdiffs(m_obj,U_data,V_data,T_prime,Q,sig_tot,sig_adv)
	
	A_gs,B_gs,C_gs,D_gs,E_gs,F_gs, J_gs, K_gs, L_gs=grid_space_transform(m_obj,U_data,V_data,vort,div,U_vert,V_vert,
		T_prime,T_prime_vert,Tbar_vert,Q,Q_vert,dlps_dmu,dlps_dlam,trip_tot,trip_adv)
	
	# do fast fourier transform
	Am, Bm, Cm, Dm, Em, Fm, Hm, Jm, Km, Lm=do_ffts(m_obj.f_obj,A_gs,B_gs,C_gs,D_gs,E_gs,F_gs,H_gs,J_gs,K_gs,L_gs)

	# do gaussian quadratures
	G_psi, G_chi, G_Tp, Emn, Fmn, Hmn, G_Q, Lmn= do_gauss_quad(m_obj,Am,Bm,Cm,Dm,Em,Fm,Hm, Jm, Km, Lm)

	# calculate the explicitly treated time derivatives for each variable
	dchi= G_chi+tf.cast(m_obj.n2/A2,np.csingle)*(Emn)
	dpsi= -G_psi
	dT = -G_Tp+Fmn
	dlps= Hmn
	dq = -G_Q + Lmn

	return dchi, dpsi, dT, dlps, dq

def add_diffusion_tend(m_obj,state,dchi, dpsi, dT, dlps, dq):
	'''
	Calculates diffusion for each variable and adds to existing tendency values

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with current spherical harmonic coefficients of each variable
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
	Returns:
		updated time derivatives, with diffusion added to the tendency
	'''
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER

	# add diffusion to the tendencies
	dchi=dchi-2*K*(tf.cast(-m_obj.n2[:]/A2,np.csingle)**order-(2/A2)**order)*state["chi_amn"]*tf.cast(-m_obj.n2[:],dtype=np.csingle)
	dpsi=dpsi-2*K*(tf.cast(-m_obj.n2[:]/A2,np.csingle)**order-(2/A2)**order)*state["psi_amn"]*tf.cast(-m_obj.n2[:],dtype=np.csingle)
	dT=dT-2*K*(tf.cast(-m_obj.n2[:]/A2,np.csingle)**order)*state["T_amn"]
	dq=dq-2*K*(tf.cast(-m_obj.n2[:]/A2,np.csingle)**order)*state["Q_amn"]

	return dchi, dpsi, dT, dlps, dq


def perform_diffusion(m_obj,state,dt):
	"""
	Perform time-split diffusion using euler backward method

	Args:
		m_obj: model object
		state: dictionary with current spherical harmonic coefficients of each variable
		dt (np.csingle): value of timestep length in seconds
	Returns:
		output: dictionary with updated spherical harmonic coefficients of each variable

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER
	
	output={}
	padding=tf.constant([[0,0],[1,0],[0,0]])

	# diffusion of specific humidity
	output["Q_amn"]=state["Q_amn"]
	diffQ=output["Q_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order))
	output["Q_amn"]=tf.concat([output["Q_amn"][:,0:1],diffQ],1)

	# diffusion of velocity potential
	output["chi_amn"]=state["chi_amn"]
	diffChi=(output["chi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	output["chi_amn"]=tf.concat([output["chi_amn"][:,0:1],diffChi],1)

	# diffusion of temperature
	output["T_amn"]=state["T_amn"]
	diffT=(output["T_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order)))
	output["T_amn"]=tf.concat([output["T_amn"][:,0:1],diffT],1)

	# np.log(PS)
	output["lps_amn"]=state["lps_amn"]

	# diffusion of	streamfunction
	output["psi_amn"]=state["psi_amn"]
	diffPsi=(output["psi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	output["psi_amn"]=tf.concat([output["psi_amn"][:,0:1],diffPsi],1)		

	output["Zs_amn"]=state["Zs_amn"]
	
	return output

def linear(m_obj,state):
	'''
	Calculate time derivative tendencies due to linear terms in spherical harmonic space

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with current spherical harmonic coefficients of each variable
		
	Returns:
		L_chi (tensor): linear time derivative of divergence spherical harmonic coeffs
		L_psi (tensor): linear time derivative of vorticity spherical harmonic coeffs
		L_T (tensor): linear time derivative of temperature spherical harmonic coeffs
		L_lps (tensor): linear time derivative of surface pressure spherical harmonic coeffs
		L_q (tensor): linear time derivative of specific humidity spherical harmonic coeffs
	'''
	# calculate virtual temperature and geopotential
	Tv=state["Q_amn"]+state["T_amn"]
	h=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,Tv[None,:,:,:])
	
	L_chi = tf.cast(m_obj.n2,np.csingle)*(h+tf.cast(R*m_obj.T_barw[:],np.csingle)*state["lps_amn"][:,:,:])/A2
	L_chi=L_chi[0]

	# get divergence from velocity potential, then calculate tendency for temperature, surface pressure
	D = -tf.cast(m_obj.n2,np.csingle)*state["chi_amn"]
	L_T = -mat_mul_2(m_obj.H_mat,D[None,:,:,:])[0]

	padding=tf.constant([[0,0],[1,0],[0,0]])
	L_lps=tf.pad(-mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])[0],padding)

	# streamfunction and specific humidity do not have any linear terms
	L_psi=0*L_T
	L_q=0*L_T

	return L_chi, L_psi, L_T, L_lps,L_q


def physics(m_obj,state):
	"""
	Calculate time derivative tendencies due to physical parameterizations

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with spherical harmonic coefficients of each variable
		
	Returns:
		dT_phys (tensor): time derivative of temperature spherical harmonic coeffs due to physics
		dq_phys (tensor): time derivative of specific humidity spherical harmonic coeffs due to physics
	"""
	
	############################### LARGE-SCALE CONDENSATION (need to define some of the constants)
	# calculate temperature, surface pressure, and specific humidity in grid space
	T=(m_obj.f_obj.eval(state["T_amn"],m_obj.f_obj.legfuncs))
	Q = m_obj.f_obj.eval(state["Q_amn"],m_obj.f_obj.legfuncs)
	lps = m_obj.f_obj.eval(state["lps_amn"],m_obj.f_obj.legfuncs)
	
	# calculate pressures, then get saturation vapor pressure and humidity
	pressure=(m_obj.sigmas[:,None,None]*tf.math.exp(lps))
	e_star=1000*0.6113*tf.math.exp(4880*(1./273.15-1./T))
	q_star= (R/461)*e_star/pressure
	mu_q=tf.constant(0.60779,dtype=np.single)
	
	Lv=2.25e6 # latent heat of vaporization
	# determine with grid cell is oversaturated
	oversat=tf.cast(tf.math.greater(Q,q_star*mu_q),dtype=np.single)

	# calculate tendency due to condensation with a 6 hour relaxation time
	dq_cond = oversat*(q_star*mu_q-Q)/tf.cast(60*60*6,dtype=np.single)
	dT_cond = oversat*(Q/mu_q-q_star)*Lv/1004/tf.cast(60*60*6,dtype=np.single)

	############################### CONVECTION (STILL DEBUGGING)
	"""
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

	dq_conv= isConv*(q_ref*mu_q-Q)/(60*60*5)
	dT_conv= isConv*(T_ref-T)/(60*60*5)
	
	dq_conv=tf.where(tf.math.is_nan(dq_conv),zeros,dq_conv)
	dT_conv=tf.where(tf.math.is_nan(dT_conv),zeros,dT_conv)

	dq_phys=m_obj.f_obj.calc_sh_coeffs(dq_cond+dq_conv*0)
	dT_phys=m_obj.f_obj.calc_sh_coeffs(dT_cond+dT_conv*0)"""
	
	############# Convert to spherical harmonic coeff
	dq_phys=m_obj.f_obj.calc_sh_coeffs(dq_cond)
	dT_phys=m_obj.f_obj.calc_sh_coeffs(dT_cond)

	return dq_phys, dT_phys


def leap(m_obj,state,old,dT,dlps,dchi,dpsi,dq,dt):
	"""
	Performs the leapfrog forward step with trapezoidal implicit terms

	Args:
		m_obj: model object
		state: dictionary with current spherical harmonic coefficients of each variable
		old:  dictionary with previous step's spherical harmonic coefficients of each variable
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
		dt (np.csingle): value of timestep length in seconds
	Returns:
		output: dictionary with updated spherical harmonic coefficients of each variable

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER
	
	output={}
	padding=tf.constant([[0,0],[1,0],[0,0]])

	output["Q_amn"]=(old["Q_amn"][:]+2*dt*dq[:])

	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,old["T_amn"][None,:,:,:])

	Q_avg = (output["Q_amn"]+old["Q_amn"])/2
	h_old = h_old +tf.cast(m_obj.T_barw,np.csingle)*mat_mul_2(m_obj.G_mat,Q_avg[None,:,:,:])

	tmp=(dt*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*old["lps_amn"][None,:,:,:])
		+dt*dt*(mat_mul_2(m_obj.G_mat,dT[None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*dlps[None,:,:,:]))/A2	
	tmp2=((-old["chi_amn"][None,:,1:,:]+dt*dchi[None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))
	
	# perform leapfrog time differencing
	output["chi_amn"]=tf.pad((-old["chi_amn"][:,1:,:]-(2*mat_mul_4(m_obj.imp_inv[:,:,1:],tmp2)))[0],padding)

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(output["chi_amn"]+old["chi_amn"])/2.

	output["T_amn"]=(tf.squeeze(old["T_amn"][:]+2*dt*(dT[:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))

	output["lps_amn"]=old["lps_amn"]+2*tf.pad(dt*(dlps[:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])))[0],padding)

	output["psi_amn"]=old["psi_amn"]-2*tf.pad(dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)	

	output["Zs_amn"]=state["Zs_amn"]
	
	return output

def euler(m_obj,state,dT,dlps,dchi,dpsi,dq,dt,imp_inv):
	"""
	Performs the euler forward step with trapezoidal implicit terms

	Args:
		m_obj: model object
		state: dictionary with current spherical harmonic coefficients of each variable
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
		dt (np.csingle): value of timestep length in seconds
	Returns:
		output: dictionary with updated spherical harmonic coefficients of each variable

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER

	output={}
	padding=tf.constant([[0,0],[1,0],[0,0]])

	output["Q_amn"]=(state["Q_amn"][:]+dt*dq[:])
	
	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,state["T_amn"][None,:,:,:])
	Q_avg=(output["Q_amn"]+state["Q_amn"])/2
	h_old = h_old +tf.cast(m_obj.T_barw,np.csingle)*mat_mul_2(m_obj.G_mat,Q_avg[None,:,:,:])

	tmp=((dt/2)*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*state["lps_amn"][None,:,:,:])
		+(dt/2)*(dt/2)*(mat_mul_2(m_obj.G_mat,dT[None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*dlps[None,:,:,:]))/A2	
	tmp2=((-state["chi_amn"][None,:,1:,:]+(dt/2)*dchi[None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))

	output["chi_amn"]=tf.pad((-state["chi_amn"][:,1:,:]-(2*mat_mul_4(imp_inv[:,:,1:],tmp2)))[0],padding)

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(state["chi_amn"]+output["chi_amn"])/2
	
	output["T_amn"]=(tf.squeeze(state["T_amn"][:]+dt*(dT[:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))
	
	output["lps_amn"]=state["lps_amn"]+tf.pad(dt*(dlps[:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])))[0],padding)

	output["psi_amn"]=state["psi_amn"]-tf.pad(dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["Zs_amn"]=state["Zs_amn"]
	return output


def euler_BE(m_obj,state,dT,dlps,dchi,dpsi,dq,dt,imp_inv):
	"""
	Performs the euler forward step with backward euler implicit terms

	Args:
		m_obj: model object
		state: dictionary with current spherical harmonic coefficients of each variable
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
		dt (np.csingle): value of timestep length in seconds
	Returns:
		output: dictionary with updated spherical harmonic coefficients of each variable

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER

	output={}
	padding=tf.constant([[0,0],[1,0],[0,0]])

	output["Q_amn"]=(state["Q_amn"][:]+dt*dq[:])

	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,state["T_amn"][None,:,:,:])
	h_old = h_old +tf.cast(m_obj.T_barw,np.csingle)*mat_mul_2(m_obj.G_mat,output["Q_amn"][None,:,:,:])

	tmp=(dt*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*state["lps_amn"][None,:,:,:])
		+dt*dt*(mat_mul_2(m_obj.G_mat,dT[None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*dlps[None,:,:,:]))/A2	
	tmp2=((-state["chi_amn"][None,:,1:,:]+dt*dchi[None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))
	
	# perform euler time differencing
	output["chi_amn"]=tf.pad((-(mat_mul_4(imp_inv[:,:,1:],tmp2)))[0],padding)

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(output["chi_amn"])

	output["T_amn"]=(tf.squeeze(state["T_amn"][:]+dt*(dT[:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))
	
	output["lps_amn"]=state["lps_amn"]+tf.pad(dt*(dlps[:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])))[0],padding)
		
	output["psi_amn"]=state["psi_amn"]-tf.pad(dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["Zs_amn"]=state["Zs_amn"]

	return output


def explicit_step(m_obj,state,dT,dlps,dchi,dpsi,dq,dt):
	"""
	Performs timestep where every term is treated explicitly

	Args:
		m_obj: model object
		state: dictionary with current spherical harmonic coefficients of each variable
		dchi (tensor): time derivative of divergence spherical harmonic coeffs
		dpsi (tensor): time derivative of vorticity spherical harmonic coeffs
		dT (tensor): time derivative of temperature spherical harmonic coeffs
		dlps (tensor): time derivative of surface pressure spherical harmonic coeffs
		dt (np.csingle): value of timestep length in seconds
	Returns:
		output: dictionary with updated spherical harmonic coefficients of each variable

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER

	output={}
	padding=tf.constant([[0,0],[1,0],[0,0]])

	output["Q_amn"]=(state["Q_amn"][:]+dt*dq[:])

	# euler forward
	output["chi_amn"]=state["chi_amn"]-tf.pad(dt*dchi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["T_amn"]=(state["T_amn"][:]+dt*dT[:])
	
	output["lps_amn"]=state["lps_amn"]+tf.pad(dt*dlps[:,1:],padding)

	output["psi_amn"]=state["psi_amn"]-tf.pad(dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["Zs_amn"]=state["Zs_amn"]

	return output