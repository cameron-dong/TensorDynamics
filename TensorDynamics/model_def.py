import numpy as np
import tensorflow as tf
import TensorDynamics.sphere_harm_new as sh

from TensorDynamics.timestepper import euler_step, leap_step
import TensorDynamics.constants as constants
from TensorDynamics.model_helpers import get_levels, get_lorenz_matrices, get_matrices, vdiff_inds


A=constants.A_EARTH #radius of earth
A2=A*A
OMEGA=constants.OMEGA



######################################################################################
class model:
	""" Python object that holds all the parameters and methods for the primitive equation model
	
	Attributes:
		f_obj: spherical harmonic object
		f (tensor): grid space values of planetary vorticity
		coslats (tensor): cos(latitudes)
		sigmas: sigma levels
		alphas: sigma level alpha values
		dsigs: sigma level thicknesses
		sigs_i: sigma level interfaces
		DIFFUSION_CONST: constant value to control diffusion strength
		DIFFUSION_ORDER: order for horizontal diffusion
		T_bar: Reference vertical temperature profile
		T_barw: As for T_bar, but with expanded dims for easier broadcasting
		n2: square of the total wavenumber matrix
		dt: timestep size, in seconds
		UV_obj: spherical harmonic object with one extra truncation
		G_mat: tensor with dimensions (levels, levels)
		H_mat: tensor with dimensions (levels, levels)
		imp_inv: tensor with dimensions (levels, levels, total_wavenumber, zonal wavenumber)
		imp_inv_eul: tensor with dimensions (levels, levels, total_wavenumber, zonal wavenumber)
		inds:
		pinds:
		ninds:
	"""

	def __init__(self,nlats: int, trunc: int, nlevels: int):
		""" Initialize a primitive equations model object, set internal parameters
		
		Args:
			nlats: number of latitudes, should be 32, 64, 96, or 128
			trunc: spherical harmonic truncation limit
			nlevels: number of vertical sigma levels
		"""

		# create a spherical harmonic grid object, then calculate planetary vorticity
		self.f_obj=sh.sh_obj(nlats,trunc)
		self.f=tf.constant(2*OMEGA*np.sin(self.f_obj.lats)[None,:,None]
				*np.ones((nlevels,len(self.f_obj.lats),len(self.f_obj.lons))), dtype=np.single)
		
		self.coslats=tf.math.cos(self.f_obj.lats[None,:,None])

		# calculate equidistant sigma values for layer midpoints, layer interfaces, layer thicknesses.
		self.sigmas, self.alphas, self.dsigs, self.sigs_i=get_levels(nlevels)

		# set diffusion constants
		if nlats==128:
			self.DIFFUSION_CONST=1*10**15 #constant for diffusive smoothing
		elif nlats==96:
			self.DIFFUSION_CONST=2*10**15 #constant for diffusive smoothing
		elif nlats==64:
			self.DIFFUSION_CONST=1*10**16 #constant for diffusive smoothing
		elif nlats==32:
			self.DIFFUSION_CONST=2*10**15 #constant for diffusive smoothing
		self.DIFFUSION_ORDER=2 # number of laplacians for diffusion

		# create tensors with average temperature of 300 K for an isobaric atmosphere
		self.T_bar=tf.constant(np.ones(nlevels)*300,dtype=np.single)
		self.T_barw=self.T_bar[:,None,None]
		
		# create placeholders for square of total wavenumber
		self.n2=tf.cast(self.f_obj.n*(self.f_obj.n+1),np.single)

		# calculate appropriate timestep size, trunc should be a multiple of 21
		self.dt=np.single(60*60*(32/nlats))

		# create grid object for U,V; requires one extra truncation
		self.UV_obj=sh.sh_obj(nlats,self.f_obj.trunc+1)

		# constant matrices required for semi-implicit method: euler/lorenz and leapfrog, respectively
		self.G_mat, self.H_mat, self.imp_inv_eul=get_lorenz_matrices(self.T_bar,self.sigmas,self.alphas,self.dsigs,self.dt,self.n2)
		self.G_mat, self.H_mat, self.imp_inv=get_matrices(self.T_bar,self.sigmas,self.alphas,self.dsigs,self.dt,self.n2)
		
		# some index arrays that are useful for the vertical diffferences
		self.inds, self.pinds, self.ninds = vdiff_inds(nlevels)
	
	def stepper(self,runtime,cstate,output_interval=None):
		"""
		Step model forward by period "runtime" from current state of the atmosphere "cstate"
		
		Args:
			runtime: Period to step forward model, in hours
			cstate (dictionary): contains current grid values of wind, temperature, surface pressure, and surface geopotential
			output_interval: frequency to save model output, in hours

		Returns:
			end_state (dictionary): updated grid values, additionally with streamfunction and velocity potential
			outputs: python list containing decoded model states with period output_interval
		"""

		# calculate number of steps, encode model state
		nsteps=int(runtime*3600/self.dt)
		cstate_amn=self.encode(cstate)
		ostate_amn=self.encode(cstate)

		# initialize outputs list
		outputs=[self.decode(cstate_amn)]


		for i in tf.range(nsteps):
			if i==0:
				euler_step(self,self.dt,cstate_amn) # initial euler step
			else:
				leap_step(self,self.dt,cstate_amn,ostate_amn) # leapfrog all other steps
			if output_interval is not None:
				if ((i+1)*self.dt/3600)%output_interval==0: # if output interval is reached, save output
					outputs.append(self.decode(cstate_amn))

		# decode final state
		end_state=self.decode(cstate_amn)

		if output_interval is not None:
			return end_state,outputs
		else:
			return end_state
	

	def encode(self,mstate):
		"""
		Given initial values of wind, temperature, surface pressure, and geopotential, calculate needed 
		spherical harmonic coefficient

		Args:
			mstate: dictionary where each key corresponds to grid space values
		Returns:
			state_amn: dictionary with needed spherical harmonic coefficients for model forecast
		"""

		# convert grid values to spherical harmonic basis
		U_amn=tf.Variable(self.UV_obj.calc_sh_coeffs(mstate["u_component_of_wind"]*self.coslats/A),trainable=False)
		V_amn=tf.Variable(self.UV_obj.calc_sh_coeffs(mstate["v_component_of_wind"]*self.coslats/A),trainable=False)
		lps_amn=tf.Variable(self.f_obj.calc_sh_coeffs(tf.math.log(mstate["surface_pressure"])),trainable=False)
		T_amn=tf.Variable(self.f_obj.calc_sh_coeffs(mstate["temperature"]),trainable=False)
		Zs_amn=tf.Variable(self.f_obj.calc_sh_coeffs(mstate["geopotential_at_surface"]),trainable=False)	

		# calculate vorticity and divergence in grid space
		u_grad_x, u_grad_y = self.UV_obj.gradient(U_amn)
		v_grad_x, v_grad_y = self.UV_obj.gradient(V_amn)
		div = A*(u_grad_x + v_grad_y)/self.coslats
		vort =  A*(v_grad_x - u_grad_y)/self.coslats

		# calculate streamfunction and velocity potential in spherical harmonic space
		vort=tf.Variable(self.f_obj.calc_sh_coeffs(vort),trainable=False)
		div=tf.Variable(self.f_obj.calc_sh_coeffs(div),trainable=False)
		psi_amn=tf.Variable(self.f_obj.inverse_laplace(vort)/A2,trainable=False)
		chi_amn=tf.Variable(self.f_obj.inverse_laplace(div)/A2,trainable=False)
		
		# create dictionary
		state_amn={"lps_amn":lps_amn,"psi_amn":psi_amn,"chi_amn":chi_amn,"T_amn":T_amn,
		"U_amn":U_amn,"V_amn":V_amn,"Zs_amn":Zs_amn}
		return state_amn
	

	def decode(self,amns):
		"""
		Given model spherical harmonic coefficients, calculate output values in grid space

		Args:
			state_amn: dictionary with spherical harmonic coefficients from model forecast
		Returns:
			mstate: dictionary where each key corresponds to grid space values
			
		"""

		# calculate U, V from current streamfunction and velocity potential coeffs
		sh.calc_UV(amns["U_amn"], amns["V_amn"],self.UV_obj.m,self.UV_obj.n,amns["psi_amn"],amns["chi_amn"],trunc=self.f_obj.trunc)

		# evaluate from spherical harmonic basis into grid space
		U=self.UV_obj.eval(amns["U_amn"],self.UV_obj.legfuncs)*A/self.coslats
		V=self.UV_obj.eval(amns["V_amn"],self.UV_obj.legfuncs)*A/self.coslats
		lps=self.f_obj.eval(amns["lps_amn"],self.f_obj.legfuncs)
		psi=self.f_obj.eval((amns["psi_amn"]),self.f_obj.legfuncs)*A2
		chi=self.f_obj.eval((amns["chi_amn"]),self.f_obj.legfuncs)*A2
		T=self.f_obj.eval(amns["T_amn"],self.f_obj.legfuncs)
		Zs=self.f_obj.eval(amns["Zs_amn"],self.f_obj.legfuncs)

		# create dictionary
		mstate={"surface_pressure":tf.math.exp(lps),"streamfunction":psi,"velocity_potential":chi,"temperature":T,
		"u_component_of_wind":U,"v_component_of_wind":V,"geopotential_at_surface":Zs}
		return mstate
