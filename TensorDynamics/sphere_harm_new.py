import tensorflow as tf
import numpy as np
from scipy.special import factorial, roots_legendre, lpmv
import TensorDynamics.constants as constants

a=constants.A_EARTH #radius of earth


def factorial_new(i):
	'''
	# new factorial definition to deal with negative inputs, "i" may be an array of numbers

	Args:
		i: np.array of values, may be positive or negative
	Returns:
		tmp: np.array of output factorials for each value in "i"
	'''
	result=factorial(np.abs(i).astype(np.double),exact=False)
	result[i<0]*=((-1.)**i[i<0])
	return result


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

	# calculate normalized values, cast back to tf tensor
	constants=(np.emath.sqrt((2.*n+1.)*factorial_new(n-m)/2./factorial_new(n+m)))
	out=constants*(lpmv(m,n,np.double(mu))/((-1)**m))

	legfuncs=tf.constant(np.single(np.real(out)),dtype=np.csingle)
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
	return fourier_coeffs

def calc_UV(U_amn,V_amn,m,n,psi_amn,chi_amn,trunc=tf.constant(42)):
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
		Spherical harmonic coefficients for U,V are updated in place

	'''
	# note that psi_amn and chi_amn should have a truncation of one less that that for U,V
	# m,n = square tensors with wavenumbers for U,V

	# cast to np.csingle to facilitate type compatibility
	m=tf.cast(m,np.csingle)
	n=tf.cast(n,np.csingle)

	# epsilon values for recursion relations
	eps_plus=(tf.math.pow((((n+1)**2-m**2)/(4*(n+1)**2-1)),0.5))
	eps_min=(tf.math.pow((((n)**2-m**2)/(4*(n)**2-1)),0.5))

	# First set to zero
	U_amn.assign(tf.zeros(tf.shape(U_amn),dtype=np.csingle))
	V_amn.assign(tf.zeros(tf.shape(V_amn),dtype=np.csingle))

	# add values corresponding to zonal derivative
	U_amn[:,:trunc+1,:trunc+1].assign(1j*m[:trunc+1,:trunc+1]*chi_amn)
	V_amn[:,:trunc+1,:trunc+1].assign(1j*m[:trunc+1,:trunc+1]*psi_amn)


	#add values for recursive relations of meridional derivative
	U_amn[:,1:,:trunc+1].assign(U_amn[:,1:,:trunc+1]+(n[1:,:trunc+1]-1)*eps_min[1:,:trunc+1]*psi_amn)
	V_amn[:,1:,:trunc+1].assign(V_amn[:,1:,:trunc+1]-(n[1:,:trunc+1]-1)*eps_min[1:,:trunc+1]*chi_amn)
	U_amn[:,:trunc,:trunc+1].assign(U_amn[:,:trunc,:trunc+1]-(n[:trunc,:trunc+1]+2)*eps_plus[:trunc,:trunc+1]*psi_amn[:,1:,:])
	V_amn[:,:trunc,:trunc+1].assign(V_amn[:,:trunc,:trunc+1]+(n[:trunc,:trunc+1]+2)*eps_plus[:trunc,:trunc+1]*chi_amn[:,1:,:])

	return None


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

		laplace_vals=amn*tf.cast(tf.cast(-self.n*(self.n+1),np.single)/a/a,np.csingle)**order
		return laplace_vals
	
	def x_deriv(self,amn):
		"""
		Calculate grid values of the zonal derivative in cartesian coordinates

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)

		Returns:
			tensor with dimensions (level, latitude, longitude)
		"""
		return self.eval(self.lambda_deriv(amn),self.legfuncs)/tf.math.cos(self.lats)[None,:,None]/a
		
	def y_deriv(self,amn):
		"""
		Calculate grid values of the meridional derivative in cartesian coordinates

		Args:
			amn:tensor with dimensions (level, zonal wavenumber, total wavenumber)

		Returns:
			tensor with dimensions (level, latitude, longitude)
		"""
		return self.eval(amn, self.phiderivs)/a
	
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
		inv_laplace_vals=tf.Variable(amn[:]*0)
		inv_laplace_vals[:,1:].assign(amn[:,1:]/tf.cast(tf.cast(-self.n[1:]*(self.n[1:]+1),np.single)/a/a,np.csingle))
		return inv_laplace_vals
