import numpy as np
import tensorflow as tf
import TensorDynamics.sphere_harm_new as sh
import TensorDynamics.constants as constants
from TensorDynamics.integration_helpers import to_grid_space, get_Gk, vertical_sums, do_vdiffs, do_ffts, do_gauss_quad, grid_space_transform, mat_mul_4, mat_mul_2

A=constants.A_EARTH #radius of earth
A2 = A*A
R=constants.DRY_AIR_CONST # gas constant


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
	vort, div, T_prime, dlps_dmu, dlps_dlam, U_data, V_data=to_grid_space(m_obj,state)

	# treatment of vertical differences and multiplication of nonlinear terms
	Gk_adv,Gk = get_Gk(m_obj.f_obj,div,dlps_dlam,U_data,dlps_dmu,V_data)
	H_gs, sig_tot,sig_adv,trip_adv,trip_tot= vertical_sums(m_obj,Gk,Gk_adv,m_obj.dsigs,m_obj.alphas,m_obj.sigmas)
	V_vert, U_vert, T_prime_vert,Tbar_vert=do_vdiffs(m_obj,U_data,V_data,T_prime,sig_tot,sig_adv)
	A_gs,B_gs,C_gs,D_gs,E_gs,F_gs=grid_space_transform(m_obj,U_data,V_data,vort,div,U_vert,V_vert,T_prime,T_prime_vert,Tbar_vert,dlps_dmu,dlps_dlam,trip_tot,trip_adv)
	
	# do fast fourier transform
	Am, Bm, Cm, Dm, Em, Fm, Hm=do_ffts(m_obj.f_obj,A_gs,B_gs,C_gs,D_gs,E_gs,F_gs,H_gs)

	# do gaussian quadratures
	G_psi, G_chi, G_Tp, Emn, Fmn, Hmn= do_gauss_quad(m_obj,Am,Bm,Cm,Dm,Em,Fm,Hm)

	# calculate the explicitly treated time derivatives for each variable
	dchi= G_chi+tf.cast(m_obj.n2/A2,np.csingle)*(Emn)
	dpsi= -G_psi
	dT = -G_Tp+Fmn
	dlps= Hmn

	return dchi, dpsi, dT, dlps


def leap(m_obj,state,old,dT,dlps,dchi,dpsi,dt):
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
		Spherical harmonic coeffs in 'state' are updated in place

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER

	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,old["T_amn"][None,:,:,:])
	tmp=(dt*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*old["lps_amn"][None,:,:,:])
		+dt*dt*(mat_mul_2(m_obj.G_mat,dT[None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*dlps[None,:,:,:]))/A2	
	tmp2=((-old["chi_amn"][None,:,1:,:]+dt*dchi[None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))
	
	# perform leapfrog time differencing and then horizontal diffusion for velocity potential
	state["chi_amn"][:,1:,:].assign((-old["chi_amn"][:,1:,:]-(2.*mat_mul_4(m_obj.imp_inv[:,:,1:],tmp2)))[0])
	state["chi_amn"][:,1:].assign(state["chi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(state["chi_amn"]+old["chi_amn"])/2.

	# leapfrog forward and diffusion of temperature
	state["T_amn"][:].assign(tf.squeeze(old["T_amn"][:]+2*dt*(dT[:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))
	#state["T_amn"][:,1:].assign(state["T_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order)))
	
	# leapfrog forward np.log(PS)
	state["lps_amn"][:,1:].assign((old["lps_amn"][:,1:]+2*dt*(dlps[:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:]))))[0])
		

	# leapfrog forward and diffusion of	streamfunction
	state["psi_amn"][:,1:].assign(tf.squeeze(old["psi_amn"][:,1:]-2*dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle)))
	state["psi_amn"][:,1:].assign(state["psi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	return None

def euler(m_obj,state,dT,dlps,dchi,dpsi,dt):
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
		Spherical harmonic coeffs in 'state' are updated in place

	"""
	K=m_obj.DIFFUSION_CONST
	order=m_obj.DIFFUSION_ORDER
	
	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,state["T_amn"][None,:,:,:])
	tmp=((dt/2)*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*state["lps_amn"][None,:,:,:])
		+(dt/2)*(dt/2)*(mat_mul_2(m_obj.G_mat,dT[None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*dlps[None,:,:,:]))/A2	
	tmp2=((-state["chi_amn"][None,:,1:,:]+(dt/2)*dchi[None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))

	D=tf.identity(state["chi_amn"])

	# perform euler time differencing and then horizontal diffusion for velocity potential
	state["chi_amn"][:,1:,:].assign((-state["chi_amn"][:,1:,:]-(2.*mat_mul_4(m_obj.imp_inv_eul[:,:,1:],tmp2)))[0])
	state["chi_amn"][:,1:].assign(state["chi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	
	# trapezoidal calculation of divergence
	
	D = -tf.cast(m_obj.n2,np.csingle)*(D+state["chi_amn"])/2.

	# euler forward and diffusion of temperature
	state["T_amn"][:].assign(tf.squeeze(state["T_amn"][:]+dt*(dT[:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))
	state["T_amn"][:,1:].assign(state["T_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order)))
	
	# euler forward np.log(PS)
	state["lps_amn"][:,1:].assign((state["lps_amn"][:,1:]+dt*(dlps[:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:]))))[0])
	
	# euler forward and diffusion of streamfunction
	state["psi_amn"][:,1:].assign(tf.squeeze(state["psi_amn"][:,1:]-dt*dpsi[:,1:]/tf.cast(m_obj.n2[1:],np.csingle)))
	state["psi_amn"][:,1:].assign(state["psi_amn"][:,1:]/(1+2*dt*K*(tf.cast(m_obj.n2[1:]/A2,np.csingle)**order-(2/A2)**order)))
	return None