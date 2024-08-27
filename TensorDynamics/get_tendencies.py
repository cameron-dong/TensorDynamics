'''Functions to calculate different time tendency terms'''

import numpy as np
import tensorflow as tf
import TensorDynamics.constants as constants
from TensorDynamics.integration_helpers import to_grid_space, get_Gk, vertical_sums, do_vdiffs, do_ffts, do_gauss_quad, grid_space_transform, mat_mul_4, mat_mul_2

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
		explicit: dictionary with spherical harmonic coefficient time tendencies of each variable
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
	explicit = {}
	explicit["chi_amn"]= G_chi+tf.cast(m_obj.n2/A2,np.csingle)*(Emn)
	explicit["psi_amn"]= -G_psi
	explicit["T_amn"] = -G_Tp+Fmn
	explicit["lps_amn"]= Hmn
	explicit["Q_amn"] = -G_Q + Lmn

	return explicit

def add_diffusion_tend(m_obj,state,dchi, dpsi, dT, dlps, dq):  #### UPDATE TO USE A DICTIONARY LIKE OTHER FUNCTIONS
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

def linear(m_obj,state):
	'''
	Calculate time derivative tendencies due to linear terms in spherical harmonic space

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with current spherical harmonic coefficients of each variable
		
	Returns:
		l_derivs: dictionary with spherical harmonic coefficient time tendencies of each variable

	'''
	# calculate virtual temperature and geopotential
	Tv=state["Q_amn"]+state["T_amn"]
	h=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,Tv[None,:,:,:])

	l_derivs={}
	
	L_chi = tf.cast(m_obj.n2,np.csingle)*(h+tf.cast(R*m_obj.T_barw[:],np.csingle)*state["lps_amn"][:,:,:])/A2
	l_derivs["chi_amn"]=L_chi[0]

	# get divergence from velocity potential, then calculate tendency for temperature, surface pressure
	D = -tf.cast(m_obj.n2,np.csingle)*state["chi_amn"]
	l_derivs["T_amn"] = -mat_mul_2(m_obj.H_mat,D[None,:,:,:])[0]

	padding=tf.constant([[0,0],[1,0],[0,0]])
	l_derivs["lps_amn"]=tf.pad(-mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])[0],padding)

	# streamfunction and specific humidity do not have any linear terms
	l_derivs["psi_amn"]=0
	l_derivs["Q_amn"]=0

	return l_derivs


def physics(m_obj,state):
	"""
	Calculate time derivative tendencies due to physical parameterizations

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with spherical harmonic coefficients of each variable
		
	Returns:
		p_derivs: dictionary with spherical harmonic coefficient time tendencies of each variable
	"""	
	
    # initiate values to zero
	p_derivs={}
	p_derivs["chi"]=0
	p_derivs["psi"]=0
	p_derivs["Q"]=0
	p_derivs["T"]=0
	p_derivs["lps"]=0
	
    # calculate needed values in grid space
	grid_state={}
	for grid_var in m_obj.physics_to_grid:
		grid_state[grid_var] = m_obj.f_obj.eval(state[grid_var+"_amn"],m_obj.f_obj.legfuncs)

    # loop through all desired parameterization functions
	for param in m_obj.param_list:
		p_derivs=param(m_obj,state,grid_state,p_derivs)

	return p_derivs

