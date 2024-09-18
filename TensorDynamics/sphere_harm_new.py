''' Functions and python object for spherical harmonic transformations'''

import tensorflow as tf
import numpy as np
from scipy.special import roots_legendre
import TensorDynamics.constants as constants
import jax

A=constants.A_EARTH #radius of earth

def leg_AS(m,n,mu):
	'''
	Calculate values of associated legendre functions with wavenumbers m,n at sin(lat)=mu

	Args:
		m: Array or tensor of zonal wavenumbers
		n: Array or tensor of total wavenumbers
		mu: Array or tensor of sin(lats)
	Returns:
		legfuncs: tf.Tensor of the orthonormalized values for each associated Legendre function
	'''

	# convert tf tensors to np arrays
	m=np.array(m)
	mu=np.array(mu)
	n=np.array(n)

	# calculate legendre function values for nonnegative integers
	max_mn=np.max([m,n])
	tmp_vals=jax.scipy.special.lpmn_values(max_mn,max_mn,mu[:,0,0],is_normalized=True)
	tmp_vals=np.moveaxis(tmp_vals,[0,1,2],[2,1,0])

	# create placeholder output array of zeros
	output=np.zeros((len(mu),len(n),len(m)))
	uniq_n=np.sort(np.unique(n)).astype(int)
	uniq_m=np.sort(np.unique(m)).astype(int)

	# insert correct values for nonnegative wavenumbers
	if np.min(n)<0:
		nstart=int(-np.min(n))
	else:
		nstart=0
	if np.min(m)<0:
		mstart=int(-np.min(m))
	else:
		mstart=0
	output[:,nstart:,mstart:]=tmp_vals[:,uniq_n[nstart]:uniq_n[-1]+1,uniq_m[mstart]:uniq_m[-1]+1]
		
	# add correct scaling for orthonormalization, correct sign	
	scaling=(1./np.sqrt(2))/jax.scipy.special.lpmn_values(0,0,np.array([0.]),is_normalized=True)
	legfuncs=tf.constant(output*scaling*(-1)**m,dtype=np.csingle)

	return legfuncs


def leg_deriv(m,n,mu):
	'''
	Calculate derivatives of associated legendre functions with wavenumbers m,n at sin(lat)=mu with
	respect to sin(lat)=mu

	Args:
		m: Array or tensor of zonal wavenumbers
		n: Array or tensor of total wavenumbers
		mu: Array or tensor of sin(lats)
	Returns:
		tf.Tensor of the derivative values for each associated legendre function with respect to mu
	'''
	# convert tf tensors to np arrays
	m=np.array(m)
	n=np.array(n)
	mu=np.array(mu)

	# Calculate latitudinal derivative using recursive relation
	eps_plus=np.emath.sqrt(((n+1)**2-m**2)/(4*(n+1)**2-1))
	eps_min=np.emath.sqrt(((n)**2-m**2)/(4*(n)**2-1))
	output=(-n*eps_plus*leg_AS(m,n+1,mu)+(n+1)*eps_min*leg_AS(m,n-1,mu))/(1-mu*mu)
	return tf.constant(np.single(np.real(output)),dtype=np.csingle)


def gauss_quad(data,weights):
	'''
	Perform gaussian quadrature along the latitude dimension (assumed axis=-3), for given data
	
	Args:
		weights: weights for the gaussian quadrature of shape (None, nlats, None, None)
		data: fourier_coeffs*legfuncs, with shape (nlevels, nlats, trunc+1, trunc+1)
			corresponding to (level, latitude, total wavenumber, zonal wavenumber)
	Returns:
		spherical harmonic coeffs: tf.Tensor with shape (nlevels, trunc+1, trunc+1)
	'''
	return tf.math.reduce_sum(data*weights,axis=-3)


def calc_am_u(data,trunc):
	'''
	Compute fourier coefficients along each latitude band

	Args:
		data: tensor with dimensions (level,latitude,longitude)
		trunc: integer truncation limit
	Returns:
		fourier_coeffs: tensor with dimensions (level, latitude, zonal wavenumber)

	'''
	fourier_coeffs=tf.signal.rfft(data)[:,:,:trunc+1]
	#fourier_coeffs=tf.signal.rfft(data)[...,:trunc+1]

	return fourier_coeffs

def calc_UV(m,n,psi_amn,chi_amn,trunc=tf.constant(42)):
	'''
	Compute spherical harmonic coefficients of U, V from coefficients
	of streamfunction and velocity potential, using recursion relations

	Args:
		U_amn: tf.Variable with dims (level,total wavenumber,zonal wavenumber), containing U coeffs
		V_amn: tf.Variable with dims (level,total wavenumber,zonal wavenumber), containing V coeffs
		m: tensor of zonal wavenumbers
		n: tensor of total wavenumbers
		psi_amn: tf.Variable with dims (level,total wavenumber,zonal wavenumber), containing streamfunc coeffs
		chi_amn: tf.Variable with dims (level,total wavenumber,zonal wavenumber), containing velopot coeffs
		trunc: integer truncation limit

	Returns:
		wind: dictionary with Spherical harmonic coefficients for U,V as tf.tensor's

	'''
	# note that psi_amn and chi_amn should have a truncation of one less that that for U,V
	# m,n = square tensors with wavenumbers for U,V

	# cast to np.csingle to facilitate type compatibility
	m=tf.cast(m,np.csingle)
	n=tf.cast(n,np.csingle)

	# epsilon values for recursion relations
	eps_plus=(tf.math.pow((((n+1)**2-m**2)/(4*(n+1)**2-1)),0.5))
	eps_min=(tf.math.pow((((n)**2-m**2)/(4*(n)**2-1)),0.5))

	# add values corresponding to zonal derivative
	pad_1=tf.constant([[0,0],[0,1],[0,1]])
	U_amn= tf.pad(1j*m[:trunc+1,:trunc+1]*chi_amn,pad_1)
	V_amn= tf.pad(1j*m[:trunc+1,:trunc+1]*psi_amn,pad_1)

	#add values for recursive relations of meridional derivative
	pad_2=tf.constant([[0,0],[1,0,],[0,1]])
	U_amn=U_amn + tf.pad((n[1:,:trunc+1]-1)*eps_min[1:,:trunc+1]*psi_amn,pad_2)
	V_amn=V_amn - tf.pad((n[1:,:trunc+1]-1)*eps_min[1:,:trunc+1]*chi_amn,pad_2)

	pad_3=tf.constant([[0,0],[0,2],[0,1]])
	U_amn=U_amn-tf.pad((n[:trunc,:trunc+1]+2)*eps_plus[:trunc,:trunc+1]*psi_amn[:,1:,:],pad_3)
	V_amn=V_amn+tf.pad((n[:trunc,:trunc+1]+2)*eps_plus[:trunc,:trunc+1]*chi_amn[:,1:,:],pad_3)


	wind={"U_amn":U_amn, "V_amn":V_amn}

	return wind


class sh_obj:
	"""
	Spherical harmonic object. Given a number of latitudes and truncation limit, facilitates 
	transformations between grid space and spectral space, as well as calculation of various
	derivatives

	Attributes:
		mu: root locations for gaussian quadrature with number of points (nlats) and locations mu = sin(lats) 
		weights: Weights at given mu for gaussian quadrature
		lats: latitudes for gaussian grid
		lons: longitudes for gaussian grid
		trunc: truncation limit for triangular truncation
		m: tensor with zonal wavenumbers
		n: tensor with total wavenumbers
		legfuncs: stored values of associated legendre functions
		legderivs: stored values of associated legendre function derivatives (w/ respect to mu)
		phiderivs: stored values of associated legendre function derivatives (w/ respect to latitude)
	"""

	def __init__(self,nlats,trunc=None):
		"""
		Given number of latitudes, creates spherical harmonic grid object;
		optionally provide truncation limit
		
		Args:
			nlats: integer number of latitudes, preferably multiple of 2
			trunc: integer truncation limit
		
		"""

		# find roots of legendre polynomials for given number of latitudes
		self.mu=roots_legendre(nlats)[0].astype(np.single)
		self.weights=(roots_legendre(nlats)[1].astype(np.csingle))
		
		# sort mu,weights from high to low
		idx=np.flip(np.argsort(self.mu))
		self.weights=tf.constant(self.weights[None,idx,None])
		self.mu=tf.constant(self.mu[idx])
	
		# calculate lats, lons
		self.lats=tf.math.asin(self.mu)
		self.lons=tf.constant(np.pi*np.arange(0,360,360/nlats/2)/180,dtype=np.single)
		
		# if not given, calculate truncation limit to minimize aliasing error
		if trunc is None:
			self.trunc=tf.math.round((len(self.lats)*2-1)/3)
		else:
			self.trunc=trunc # truncation limit

		# create 2-d arrays with all the possible m,n combinations in a triangular truncation
		m=np.empty(((trunc+1),(trunc+1)))
		n=np.empty(((trunc+1),(trunc+1)))
		for i in range(trunc+1):
			for j in range(trunc+1):
				m[i,j]=j
				n[i,j]=i
		self.m=tf.constant(m,dtype=np.int32)
		self.n=tf.constant(n,dtype=np.int32)

		# precompute associated legendre values for usage later
		self.legfuncs=leg_AS(self.m,self.n,self.mu[:,None,None])
		self.legderivs=leg_deriv(self.m,self.n,self.mu[:,None,None])
		self.phiderivs=self.phi_deriv()


	def calc_sh_coeffs(self,data):
		"""
		Given data on gaussian grid, calculate spherical harmonics

		Args:
			data: tensor with dimensions (level, latitude, longitude)
		Returns:
			amn: tensor with dimensions (level, zonal wavenumber, total wavenumber)

		"""
	
		# compute fourier transform along longitude
		am_u=calc_am_u(data,self.trunc)

		# compute gaussian quadrature
		amn=gauss_quad(am_u[:,:,None,:]*(self.legfuncs[None,:,:,:]),self.weights[:,:,:,None])
		#amn=gauss_quad(am_u[...,:,None,:]*(self.legfuncs[:,:,:]),self.weights[:,:,:,None])

		return amn

	def eval(self,amn,legfuncs):
		"""
		Transform spherical harmonic coefficients into grid space

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
			legfuncs: tf.Tensor of associated legendre functions
		Returns:
			output: tensor with evaluated values on a gaussian grid 
		"""

		output=(tf.signal.irfft(tf.math.reduce_sum((legfuncs*amn[:,None,:,:]),axis=-2),fft_length=tf.shape(self.lons)))
		#output=(tf.signal.irfft(tf.math.reduce_sum((legfuncs*amn[...,None,:,:]),axis=-2),fft_length=tf.shape(self.lons)))
		return output


	def phi_deriv(self):
		# Calculates derivative of associated legendre function with respect to latitude (phi)
		return (self.legderivs*tf.cast(tf.math.cos(self.lats)[:,None,None],np.csingle))


	def lambda_deriv(self,amn):
		"""
		Calculate spherical harmonic coefficients corresponding to the zonal derivative

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			lambda_deriv: tensor with dimensions (level, zonal wavenumber, total wavenumber)
		"""

		lambda_deriv=amn*1j*tf.cast(self.m,np.csingle)
		return lambda_deriv

	def laplacian(self,amn,order=1):
		"""
		Calculate spherical harmonic coefficients corresponding to the laplacian
		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			laplace_vals: tensor with dimensions (level, zonal wavenumber, total wavenumber)
		"""

		laplace_vals=amn*tf.cast(tf.cast(-self.n*(self.n+1),np.single)/A/A,np.csingle)**order
		return laplace_vals
	
	def x_deriv(self,amn):
		"""
		Calculate grid values of the zonal derivative in cartesian coordinates

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			tensor with dimensions (level, latitude, longitude)
		"""
		return self.eval(self.lambda_deriv(amn),self.legfuncs)/tf.math.cos(self.lats)[None,:,None]/A
		
	def y_deriv(self,amn):
		"""
		Calculate grid values of the meridional derivative in cartesian coordinates

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			tensor with dimensions (level, latitude, longitude)
		"""
		return self.eval(amn, self.phiderivs)/A
	
	def gradient(self,amn):
		"""
		Returns the gradient in cartesian coordinates

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			x_deriv_vals: tensor with dimensions (level, latitude, longitude)
			y_deriv_vals: tensor with dimensions (level, latitude, longitude)
		"""
		x_deriv_vals= self.x_deriv(amn)
		y_deriv_vals= self.y_deriv(amn)
		return x_deriv_vals, y_deriv_vals
	
	def inverse_laplace(self, amn):
		"""
		Calculate spherical harmonic coefficients corresponding to the inverse laplacian
		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)
		Returns:
			inv_laplace_vals: tensor with dimensions (level, zonal wavenumber, total wavenumber)
		"""
		newvals=amn[:,1:]/tf.cast(tf.cast(-self.n[1:]*(self.n[1:]+1),np.single)/A/A,np.csingle)
		padding=tf.constant([[0,0],[1,0],[0,0]])
		inv_laplace_vals=tf.pad(newvals,padding)
		return inv_laplace_vals
