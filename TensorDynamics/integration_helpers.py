
import numpy as np

import time as time
import tensorflow as tf

import TensorDynamics.sphere_harm_new as sh
import TensorDynamics.constants as constants
A=constants.A_EARTH
A2=A*A
R=constants.DRY_AIR_CONST # gas constant
kb=constants.KAPPA # scaled gas constant by specific heat


def mat_mul_4(matB,matA):
	"""
	Performs matrix multiplication for 4-D tensors whose last 2 dimensions are (total wavenumber, zonal wavenumber).
	wavenumber dimensions are treated as batch dimensions

	Args:
		matB: tensor with dimensions (level, level, total wavenumber, zonal wavenumber)
		matA: tensor with dimensions (None, level, total wavenumber, zonal wavenumber)
	Returns:
		tensor with dimensions (None, level, total wavenumber, zonal wavenumber)
	"""
	return tf.transpose(tf.linalg.matmul(tf.cast(tf.transpose(matB,[2,3,0,1]),np.csingle),tf.transpose(matA,[2,3,1,0])),[3,2,0,1])
def mat_mul_2(matB,matA):
	"""
	Performs matrix multiplication for 2-D tensor with 4-D tensor

	Args:
		matB: tensor with dimensions (level, level)
		matA: tensor with dimensions (None, level, total wavenumber, zonal wavenumber)
	Returns:
		tensor with dimensions (None, level, total wavenumber, zonal wavenumber)
	"""
	return tf.transpose(tf.linalg.matmul(tf.cast(tf.transpose(matB,[0,1]),np.csingle),tf.transpose(matA,[2,3,1,0])),[3,2,0,1])


def to_grid_space(m_obj,state):
	"""
	Calculates the explicit portions of the time derivative for each variable

	Args:
		m_obj: model object, which contains parameters of the model
		state: dictionary with current spherical harmonic coefficients of each variable
	Returns:
		vorticity (tensor): grid space evaluation of vorticity
		divergence (tensor): grid space evaluation of divergence
		T_prime (tensor): grid space deviation from isobaric basic state temperature
		dlps_dmu (tensor): grid space evaluation of meridional derivative of ln(Ps)
		dlps_dlam (tensor): grid space evaluation of zonal derivative of ln(Ps)
		U_data (tensor): grid space evaluation of (zonal wind)*(cos[lats])
		V_data (tensor): grid space evaluation of (meridional wind)*(cos[lats])
	
	"""
	f_obj=m_obj.f_obj


	# use spherical harmonic laplacian operator to calculate divergence, vorticity
	vorticity=(f_obj.eval(f_obj.laplacian(state["psi_amn"]),f_obj.legfuncs)*A2)
	divergence=(f_obj.eval(f_obj.laplacian(state["chi_amn"]),f_obj.legfuncs)*A2)


	# calculate temperature in grid space, then subtract basic state temperature
	T=(f_obj.eval(state["T_amn"],f_obj.legfuncs))
	T_prime=(T-m_obj.T_barw)

	Q = f_obj.eval(state["Q_amn"],f_obj.legfuncs)
	lps = f_obj.eval(state["lps_amn"],f_obj.legfuncs)

	# calculate derivatives of log(Ps) with respect to mu and lambda
	dlps_dmu=(f_obj.eval(state["lps_amn"],f_obj.legderivs))
	dlps_dlam=(f_obj.eval(f_obj.lambda_deriv(state["lps_amn"]),f_obj.legfuncs))

	# first calculate U, V spectral coefficients, then evaluate to grid space
	wind = sh.calc_UV(m_obj.UV_obj.m,m_obj.UV_obj.n,state["psi_amn"],state["chi_amn"],trunc=m_obj.f_obj.trunc)
	U_data=(m_obj.UV_obj.eval(wind["U_amn"],m_obj.UV_obj.legfuncs)*A)
	V_data=(m_obj.UV_obj.eval(wind["V_amn"],m_obj.UV_obj.legfuncs)*A)
	return  vorticity, divergence, T_prime, dlps_dmu, dlps_dlam, U_data, V_data, Q


def get_Gk(sh_obj,divergence,dlps_dlam,U_data,dlps_dmu,V_data):
	"""
	On each sigma midpoint level, calculate the divergence and advection of ln(Ps)
	Args:
		sh_obj: spherical harmonic grid object
		divergence (tensor): grid space evaluation of divergence
		dlps_dmu (tensor): grid space evaluation of meridional derivative of ln(Ps)
		dlps_dlam (tensor): grid space evaluation of zonal derivative of ln(Ps)
		U_data (tensor): grid space evaluation of (zonal wind)*(cos[lats])
		V_data (tensor): grid space evaluation of (meridional wind)*(cos[lats])
	Returns:
		Gk_adv (tensor): grid values of advection of ln(Ps)
		Gk (tensor): Gk_adv plus the divergence
	"""
	Gk_adv=(dlps_dlam*U_data/A/tf.math.cos(sh_obj.lats)[None,:,None]**2+dlps_dmu*V_data/A)
	Gk=(divergence+Gk_adv)
	return Gk_adv,Gk


def vertical_sums(m_obj,Gk,Gk_adv,dsigs,alphas,sigmas):
	"""
	Calculates rate of change of sigma at level interfaces, as well as the triple product of ω/σ/Ps on level midpoints
	Args:
		m_obj: model object
		Gk_adv (tensor): grid values of advection of ln(Ps)
		Gk (tensor): Gk_adv plus the divergence
		dsigs (tensor): thickness of each sigma level
		alphas (tensor): natural logs of sigma_{k+1}/sigma_{k}
		sigmas (tensor): sigma values at level midpoints
	Returns:
		total_adv_sum (tensor): sum of advection of ln(Ps) over all sigma levels
		sig_tot (tensor): sigma velocities on level interfaces due to both divergence and advection
		sig_adv(tensor): sigma velocities on level interfaces due solely to advection
		trip_tot (tensor): triple product values on level midpoints due to both divergence and advection
		trip_adv(tensor): triple product values on level midpoints due solely to advection

	"""

	# create placeholder TensorArrays for Gk sums
	Gk_sums = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas)+1, element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)
	Gk_adv_sums = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas)+1, element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)

	# calculate sums up to each interface, then convert TensorArray to Tensor
	for k in tf.range(1,len(sigmas)+1):
		Gk_sums=Gk_sums.write(k,tf.math.reduce_sum(Gk[0:k,:,:]*dsigs[0:k,None,None],axis=0))#.mark_used()
		Gk_adv_sums=Gk_adv_sums.write(k,tf.math.reduce_sum(Gk_adv[0:k,:,:]*dsigs[0:k,None,None],axis=0))#.mark_used()
	Gk_sums=Gk_sums.stack()	
	Gk_adv_sums=Gk_adv_sums.stack()
	

	# create placeholder TensorArrays for sigma velocities and triple products
	sig_tot = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas)+1, element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)
	sig_adv = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas)+1, element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)
	trip_adv = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas), element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)
	trip_tot = tf.TensorArray(dtype=tf.float32, size=len(m_obj.sigmas), element_shape=(len(m_obj.f_obj.lats),len(m_obj.f_obj.lons)), dynamic_size=False)
	
	# calculate sigma velocities and triple products at each interface or midpoint, then convert TensorArray to Tensor
	for k in tf.range(len(sigmas)):
		sig_tot =sig_tot.write(k,tf.math.reduce_sum(dsigs[0:k])*Gk_sums[len(sigmas)]-Gk_sums[k])#.mark_used()
		sig_adv = sig_adv.write(k,tf.math.reduce_sum(dsigs[0:k])*Gk_adv_sums[len(sigmas)]-Gk_adv_sums[k])#.mark_used()
		trip_adv =trip_adv.write(k,Gk_adv[k]-alphas[k]/dsigs[k]*Gk_adv_sums[k+1]-alphas[k-1]/dsigs[k]*Gk_adv_sums[k])#.mark_used()
		trip_tot = trip_tot.write(k,Gk_adv[k]-alphas[k]/dsigs[k]*Gk_sums[k+1]-alphas[k-1]/dsigs[k]*Gk_sums[k])#.mark_used()
	sig_tot=sig_tot.stack()
	sig_adv=sig_adv.stack()
	trip_adv=trip_adv.stack()
	trip_tot=trip_tot.stack()

	# Sum over the whole column of the advective term
	total_adv_sum=-Gk_adv_sums[-1,:,:][None,:,:]

	return total_adv_sum, sig_tot,sig_adv,trip_adv,trip_tot

# placeholder function for tf.gather, to save space
def tfg(a,i):
	return tf.gather(a,indices=i)


def vertical_diff(m_obj,data,sig):
	"""
	Calculates vertical advection terms using finite differences, 
	using data at midpoints and vertical velocity at interfaces

	Args:
		m_obj: model object
		data (tensor): gridpoint data at level midpoints
		sig (tensor): vertical velocity at level interfaces
	Returns:
		(tensor): vertical advection values at level midpoints
	"""

	# index arrays that simplify the syntax
	inds=m_obj.inds
	pinds=m_obj.pinds
	ninds=m_obj.ninds

	return (sig[:-1]*(data-tfg(data,ninds))/(m_obj.dsigs[:,None,None]+tfg(m_obj.dsigs,ninds)[:,None,None])
			+sig[1:]*(tfg(data,pinds)-data)/(tfg(m_obj.dsigs,pinds)[:,None,None]+m_obj.dsigs[:,None,None]))


def do_vdiffs(m_obj,U_data,V_data,T_prime,Q,sig_tot,sig_adv):
	"""
	Calculate the vertical advection terms for various variables
	Args:
		m_obj: model object
		U_data (tensor): grid space evaluation of (zonal wind)*(cos[lats])
		V_data (tensor): grid space evaluation of (meridional wind)*(cos[lats])
		T_prime (tensor): grid space deviation from isobaric basic state temperature
		sig_tot (tensor): sigma velocities on level interfaces due to both divergence and advection
		sig_adv(tensor): sigma velocities on level interfaces due solely to advection
	Returns:
		V_vert (tensor): Vertical advection term of (v wind)*(cos[lats]) at level midpoints
		U_vert (tensor): Vertical advection term of (u wind)*(cos[lats]) at level midpoints
		T_prime_vert (tensor): Vertical advection term of temperature deviations at level midpoints
		Tbar_vert (tensor): Vertical advection term of temperature basic state at level midpoints
	"""

	# vertical advection due to total vertical velocity
	V_vert=(vertical_diff(m_obj,V_data,sig_tot))
	U_vert=(vertical_diff(m_obj,U_data,sig_tot))
	T_prime_vert=(vertical_diff(m_obj,T_prime,sig_tot))
	Q_vert=(vertical_diff(m_obj,Q,sig_tot))

	# vertical advection due to vertical velocity calculated solely from horizontal velocity
	Tbar_vert=(vertical_diff(m_obj,m_obj.T_barw,sig_adv))

	return V_vert, U_vert, T_prime_vert,Tbar_vert, Q_vert


def grid_space_transform(m_obj,U_data,V_data,vorticity,divergence,U_vert,V_vert,T_prime,T_prime_vert,Tbar_vert,Q,Q_vert,dlps_dmu,dlps_dlam,trip_tot,trip_adv):
	"""
	Calculate product of nonlinear terms in grid space

	Args:
		m_obj: model object
		U_data (tensor): grid space evaluation of (zonal wind)*(cos[lats])
		V_data (tensor): grid space evaluation of (meridional wind)*(cos[lats])
		vorticity (tensor): grid space evaluation of vorticity
		divergence (tensor): grid space evaluation of divergence
		V_vert (tensor): Vertical advection term of (v wind)*(cos[lats]) at level midpoints
		U_vert (tensor): Vertical advection term of (u wind)*(cos[lats]) at level midpoints
		T_prime (tensor): grid space deviation from isobaric basic state temperature
		T_prime_vert (tensor): Vertical advection term of temperature deviations at level midpoints
		Tbar_vert (tensor): Vertical advection term of temperature basic state at level midpoints
		dlps_dmu (tensor): grid space evaluation of meridional derivative of ln(Ps)
		dlps_dlam (tensor): grid space evaluation of zonal derivative of ln(Ps)
		trip_tot (tensor): triple product values on level midpoints due to both divergence and advection
		trip_adv(tensor): triple product values on level midpoints due solely to advection
	Returns:
		Various tensors representing the nonlinear products in grid space
	"""

	Tv_prime = T_prime + (T_prime+m_obj.T_barw)*Q

	#############

	A_gs=(U_data*(vorticity+m_obj.f)+V_vert+R*Tv_prime/A*dlps_dmu*m_obj.coslats**2)
	B_gs=(V_data*(vorticity+m_obj.f)-U_vert-R*Tv_prime/A*dlps_dlam)
	C_gs=(U_data*T_prime)
	D_gs=(V_data*T_prime)
	E_gs=((U_data*U_data+V_data*V_data)/2.)
	F_gs=(T_prime*divergence - T_prime_vert -Tbar_vert +kb*(Tv_prime*trip_tot +m_obj.T_barw[:]*trip_adv))

	J_gs= U_data*Q
	K_gs= V_data*Q
	L_gs= Q*divergence - Q_vert

	return A_gs,B_gs,C_gs,D_gs,E_gs,F_gs, J_gs, K_gs, L_gs

def do_ffts(sh_obj,A_gs,B_gs,C_gs,D_gs,E_gs,F_gs,H_gs, J_gs, K_gs, L_gs):
	"""
	Calculate zonal fourier coefficients of nonlinear terms

	Args:
		sh_obj: spherical harmonic object
		Various tensors representing the nonlinear products in grid space
	Returns:
		Various tensors representing the fourier coefficients at each latitude
		with dimensions (levels, latitudes, zonal wavenumber)
	"""
	Am=(sh.calc_am_u(A_gs,sh_obj.trunc))
	Bm=(sh.calc_am_u(B_gs,sh_obj.trunc))
	Cm=(sh.calc_am_u(C_gs,sh_obj.trunc))
	Dm=(sh.calc_am_u(D_gs,sh_obj.trunc))
	Em=(sh.calc_am_u(E_gs,sh_obj.trunc))
	Fm=(sh.calc_am_u(F_gs,sh_obj.trunc))
	Hm=(sh.calc_am_u(H_gs,sh_obj.trunc))

	Jm=(sh.calc_am_u(J_gs,sh_obj.trunc))
	Km=(sh.calc_am_u(K_gs,sh_obj.trunc))
	Lm=(sh.calc_am_u(L_gs,sh_obj.trunc))
	return Am, Bm, Cm, Dm, Em, Fm, Hm, Jm, Km, Lm

def G(Rm,Sm,sh_obj):
	"""
	A useful gaussian quadrature calculation for results of a divergence or curl operation on a vector

	Args:
		Rm (tensor): fourier coefficients with dims (level, latitude, zonal wavenumber)
		Sm (tensor): fourier coefficients with dims (level, latitude, zonal wavenumber)
		sh_obj: spherical harmonic object
	Returns:
		spherical harmonic coefficients with dims (level, total wavenumber, zonal wavenumber)

	"""
	return (1./A)*sh.gauss_quad(sh_obj.legfuncs[None,:,:,:]*1j*tf.cast(sh_obj.m[None,None,:,:],np.csingle)*Rm[:,:,None,:]
		/(tf.cast(1.,np.csingle)-tf.cast(sh_obj.mu[None,:,None,None],np.csingle)**2)-Sm[:,:,None,:]*sh_obj.legderivs[None,:,:,:],sh_obj.weights[:,:,:,None])


def do_gauss_quad(m_obj,Am,Bm,Cm,Dm,Em,Fm,Hm, Jm, Km, Lm):
	"""
	Perform (meridional) gaussian quadrature of the fourier coefficients

	Args:
		m_obj: model object
		Various tensors representing the fourier coefficients at each latitude
		with dimensions (levels, latitudes, zonal wavenumber)
	Returns:
		Various tensors (levels, latitudes, total wavenumber, zonal wavenumber)
		containing spherical harmonic coefficients that
		are needed to calculate time derivatives
	"""
	f_obj=m_obj.f_obj
	G_psi=(G(Am,Bm,f_obj))
	
	G_chi=(G(Bm,-Am,f_obj))
	G_T_prime=(G(Cm,Dm,f_obj))
	Emn=(sh.gauss_quad(Em[:,:,None,:]*f_obj.legfuncs[None,:,:,:]
		/(1-tf.cast(f_obj.mu[None,:,None,None]**2,np.csingle)),f_obj.weights[:,:,:,None]))
		
	Fmn=(sh.gauss_quad(Fm[:,:,None,:]*f_obj.legfuncs[None,:,:,:],f_obj.weights[:,:,:,None]))
	Hmn=(sh.gauss_quad(Hm[:,:,None,:]*f_obj.legfuncs[None,:,:,:],f_obj.weights[:,:,:,None]))
	Lmn=(sh.gauss_quad(Lm[:,:,None,:]*f_obj.legfuncs[None,:,:,:],f_obj.weights[:,:,:,None]))

	G_Q = (G(Jm,Km,f_obj))
	
	return G_psi, G_chi, G_T_prime, Emn, Fmn, Hmn, G_Q, Lmn

