''' Functions that perform a single time step, given appropriate input tendencies'''

import numpy as np
import tensorflow as tf
import TensorDynamics.constants as constants
from TensorDynamics.integration_helpers import mat_mul_4, mat_mul_2


A=constants.A_EARTH #radius of earth
A2 = A*A
R=constants.DRY_AIR_CONST # gas constant
KAPPA=constants.KAPPA

def perform_diffusion(m_obj,state,dt):
	"""
	Perform time-split diffusion using euler backward method over period 2*dt

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

def leap(m_obj,state,old,derivs,dt):
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

	output["Q_amn"]=(old["Q_amn"][:]+2*dt*derivs["Q_amn"])

	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,old["T_amn"][None,:,:,:])

	Q_avg = (output["Q_amn"]+old["Q_amn"])/2
	h_old = h_old +tf.cast(m_obj.T_barw,np.csingle)*mat_mul_2(m_obj.G_mat,Q_avg[None,:,:,:])

	tmp=(dt*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*old["lps_amn"][None,:,:,:])
		+dt*dt*(mat_mul_2(m_obj.G_mat,derivs["T_amn"][None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*derivs["lps_amn"][None,:,:,:]))/A2	
	tmp2=((-old["chi_amn"][None,:,1:,:]+dt*derivs["chi_amn"][None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))
	
	# perform leapfrog time differencing
	output["chi_amn"]=tf.pad((-old["chi_amn"][:,1:,:]-(2*mat_mul_4(m_obj.imp_inv[:,:,1:],tmp2)))[0],padding)

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(output["chi_amn"]+old["chi_amn"])/2.

	output["T_amn"]=(tf.squeeze(old["T_amn"][:]+2*dt*(derivs["T_amn"][:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))

	output["lps_amn"]=old["lps_amn"]+2*tf.pad(dt*(derivs["lps_amn"][:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])))[0],padding)

	output["psi_amn"]=old["psi_amn"]-2*tf.pad(dt*derivs["psi_amn"][:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)	

	output["Zs_amn"]=state["Zs_amn"]
	
	return output

def euler(m_obj,state,derivs,dt,imp_inv):
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


def euler_BE(m_obj,state,derivs,dt,imp_inv):
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

	output["Q_amn"]=(state["Q_amn"][:]+dt*derivs["Q_amn"][:])

	# calculation of terms for evolving velocity potential
	h_old=state["Zs_amn"][None,:,:,:]+mat_mul_2(m_obj.G_mat,state["T_amn"][None,:,:,:])
	h_old = h_old +tf.cast(m_obj.T_barw,np.csingle)*mat_mul_2(m_obj.G_mat,output["Q_amn"][None,:,:,:])

	tmp=(dt*(h_old+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*state["lps_amn"][None,:,:,:])
		+dt*dt*(mat_mul_2(m_obj.G_mat,derivs["T_amn"][None,:,:,:])+tf.cast(R*m_obj.T_barw[None,:],np.csingle)*derivs["lps_amn"][None,:,:,:]))/A2	
	tmp2=((-state["chi_amn"][None,:,1:,:]+dt*derivs["chi_amn"][None,:,1:,:]/tf.cast(m_obj.n2[1:],np.csingle)+tmp[:,:,1:]))
	
	# perform euler time differencing
	output["chi_amn"]=tf.pad((-(mat_mul_4(imp_inv[:,:,1:],tmp2)))[0],padding)

	# trapezoidal calculation of divergence
	D = -tf.cast(m_obj.n2,np.csingle)*(output["chi_amn"])

	output["T_amn"]=(tf.squeeze(state["T_amn"][:]+dt*(derivs["T_amn"][:]-(mat_mul_2(m_obj.H_mat,D[None,:,:,:])))))
	
	output["lps_amn"]=state["lps_amn"]+tf.pad(dt*(derivs["lps_amn"][:,1:]-(mat_mul_2(m_obj.dsigs[None,:],D[None,:,1:,:])))[0],padding)
		
	output["psi_amn"]=state["psi_amn"]-tf.pad(dt*derivs["psi_amn"][:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["Zs_amn"]=state["Zs_amn"]

	return output


def explicit_step(m_obj,state,derivs,dt):
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

	output["Q_amn"]=(state["Q_amn"][:]+dt*derivs["Q_amn"][:])

	# euler forward
	output["chi_amn"]=state["chi_amn"]-tf.pad(dt*derivs["chi_amn"][:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["T_amn"]=(state["T_amn"][:]+dt*derivs["T_amn"][:])
	
	output["lps_amn"]=state["lps_amn"]+tf.pad(dt*derivs["lps_amn"][:,1:],padding)

	output["psi_amn"]=state["psi_amn"]-tf.pad(dt*derivs["psi_amn"][:,1:]/tf.cast(m_obj.n2[1:],np.csingle),padding)

	output["Zs_amn"]=state["Zs_amn"]

	return output