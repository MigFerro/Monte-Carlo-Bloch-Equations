#!/usr/bin/env python

'''
Functions used for simulation the field inside the laser cavity:
	- gaussian
	- rectangular impulse
	- cavity field (simple, no Doppler)
	- wave with frequency changes
	- cavity field with Doppler shifts

All functions make use of the Numba library to improve performance

'''

# standard library
import numpy as np

# Numba
from numba import njit, prange, vectorize

# local source library
import constants as ct
import maxwell as mx
import fileIO as fIO



@vectorize
def unit_field(t):

	'''
	Unitary field for all times (for testing)

	Input

		t - time value

	Return

		1.0

	'''

	return 1.0



# pulse shapes



@vectorize
def rect(t, t0, tau):

	'''
	Rectangular impulse

	Input

		t - time value
		t0 - center of the impulse
		tau - duration of the impulse

	Return

		rectangle impulse function value

	'''

	if np.abs(t-t0) > tau/2:
		return 0.0
	else:
		return 1.0



@njit
def normConst_rectSq(tau):

	'''
	Normalizing constant for Integral( Rect^2 ) = 1

	Input

		tau - duration of the rectangle impulse

	Return

		1/sqrt(tau)

	'''

	return 1/np.sqrt(tau)



@vectorize
def gauss(t, t0, tau):

	'''
	Gausian pulse

	Input

		t - time value
		t0 - center of the Gaussian pulse
		tau - width (standard deviation) of the Gaussian pulse

	Return

		Gaussian pulse value at t

	'''

	if np.abs(t-t0) > 5*tau:
		return 0.0
	else:
		x = (t - t0)/tau
		return np.exp(-(1/2)*x**2)



@njit
def normConst_gaussSq(tau):

	'''
	Normalizing constant such that Integral( Gauss^2 ) = 1.0

	Input

		tau - width (standard deviation) of the Gaussian pulse

	Return

		normalizing constant = 1/sqrt( tau* sqrt( pi ) )

	'''

	return (1/np.sqrt(tau*np.sqrt(np.pi)))



# time array considerations


@njit
def t_maxn(tau, t0, D, R, minAmp):

	'''
	Calculates the time needed for a cavity field

	Input

		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)

	Return

		t - numpy array with the time values
		maxN - maximum reflection to calculate according to minAmp

	'''
	
	maxN = 1 + int(np.log(minAmp)/np.log(R))


	# this was determined by looking at several test results
	# more time is needed for short duration pulses so a "time cushion" is added
	if tau < 15E-9:
		t_cushion = 300E-9
	else:
		t_cushion = 0.0

	if tau <= 100.0E-9:
		tmax = t0 + maxN*(D/ct.c) + 10*tau + t_cushion
	else:
		tmax = t0 + maxN*(D/ct.c) + 5*tau + t_cushion

	# for the cavity field without Doppler we consider this sampling frequency because we only need to
	# sample the Rabi frequency which is of the order ~ 10^5 - 10^7 and tau is generally of the order ~10^-9 - 10^-7 

	dt = tau/10

	t = np.arange(0.0, tmax, dt)

	return t, maxN



@njit
def t_maxn_dopp(tau, t0, D, R, minAmp):

	'''
	Calculates the time needed for a Doppler-shifted cavity field

	Input

		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)

	Return

		t - numpy array with the time values
		maxN - maximum reflection to calculate according to minAmp

	'''
	
	maxN = 1 + int(np.log(minAmp)/np.log(R))

	if tau < 15E-9:
		t_cushion = 300E-9
	else:
		t_cushion = 0.0

	# the values 10*tau and 5*tau where determined by analysing several test results
	if tau <= 100.0E-9:
		tmax = t0 + maxN*(D/ct.c) + 10*tau + t_cushion
	else:
		tmax = t0 + maxN*(D/ct.c) + 5*tau + t_cushion

	dt = 1.0E-9 #limit for sampling of Doppler frequency f*v/c (assuming v <= 1000 m/s)

	t = np.arange(0.0, tmax, dt)

	return t, maxN



# cavity field with no Doppler effect



@njit
def cavityField(tau, t0, D, R, minAmp):

	'''
	Simulates the electric field inside the laser cavity (no Doppler)

	Input

		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)	

	Return 

		field - numpy array with the complex valued cavity field amplitude
		t - numpy array with the time values (from t_maxn function)
		maxN - max reflection calculated (from t_maxn function)

	'''

	t, maxN = t_maxn(tau, t0, D, R, minAmp)

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*np.exp(1j*phi[i])

	field = normConst_gaussSq(tau)*field

	return field, t, maxN



@njit
def cavityField_rawCycle(t, tau, t0, D, R, maxN):

	'''
	Similar function to cavityField but takes t and maxN as input
	to not re-calculate these values when used in a cycle

	Input

		t - numpy array with time values
		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		maxN - maximum reflection to consider	

	Return 

		field - numpy array with the complex valued cavity field amplitude
	
	'''

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*np.exp(1j*phi[i])

	field = normConst_gaussSq(tau)*field

	return field



@njit(parallel=True)
def simul_fields(nsim, tau, D, R):

	'''
	Generates several cavity fields in a cycle (parallelized with Numba)

	Input

		nsim - number of fields to simulate
		tau - duration of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance

	Return

		t - numpy array with the time values used
		Efields_arr - numpy array with all the simulated fields

	'''

	Efield, t, maxN = cavityField(tau, 4*tau, D, R, 0.01)

	Efields_arr = np.zeros(shape=(nsim, len(t)), dtype=np.csingle)

	Efields_arr[0,:] = Efield

	for i in prange(1, nsim):

		Efields_arr[i,:] = cavityField_rawCycle(t, tau, 4*tau, D, R, maxN)

	return t, Efields_arr



# Doppler with new velocity each field reflection
# to use with decay rates in the Bloch equations



@njit
def cavityField_doppVel(tau, t0, D, R, m, T, carrier_freq, minAmp):

	'''
	Simulates the electric field inside the laser cavity
	Doppler effect included with new velocity sampled at each reflection

	Input

		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)
		m - mass of the particle
		T - temperature in K
		carrier_freq - frequency of the laser	

	Return 

		field - numpy array with the complex valued Doppler shifted cavity field amplitude
		t - numpy array with the time values (from t_maxn function)
		maxN - max reflection calculated (from t_maxn function)

	'''

	t, maxN = t_maxn_dopp(tau, t0, D, R, minAmp)

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	v = mx.MB_velocity(m, T, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*np.exp(1j*carrier_freq*v[i]/ct.c + 1j*phi[i])

	field = normConst_gaussSq(tau)*field

	return field, t, maxN



@njit
def cavityField_doppVel_rawCycle(t, tau, t0, D, R, m, T, carrier_freq, maxN):

	'''
	Similar function to cavityField_doppVel but takes as input t and maxN

	Input

		t - numpy array with the time values
		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)	
		m - mass of the particle
		T - temperature
		carrier_freq - frequency of the laser
		maxN - maximum relfection to consider

	Return 

		field - numpy array with the complex valued Doppler shifted cavity field amplitude

	'''

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	v = mx.MB_velocity(m, T, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*np.exp(1j*carrier_freq*v[i]/ct.c + 1j*phi[i])

	field = normConst_gaussSq(tau)*field

	return field, t, maxN



@njit(parallel=True)
def simul_fields_doppVel(nsim, tau, D, R, m, T, carrier_freq, minAmp=0.01):

	'''
	Generates several Doppler shifted cavity fields in a cycle (parallelized with Numba)
	New velocity sampled at each reflection

	Input

		nsim - number of fields to simulate
		tau - duration of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		m - mass of the particle
		T - temperature
		collision_rate - collision rate
		carrier_freq - frequency of the laser

	Return

		t - numpy array with the time values used
		Efields_arr - numpy array with all the simulated fields

	'''

	Efield_dopp, t, maxN = cavityField_doppVel(tau, 4*tau, D, R, m, T, carrier_freq, minAmp)

	Efields_dopp_arr = np.zeros(shape=(nsim, len(t)), dtype=np.csingle)

	Efields_dopp_arr[0,:] = Efield_dopp

	for i in prange(1, nsim):

		Efields_dopp_arr[i,:] = cavityField_doppVel_rawCycle(t, tau, 4*tau, D, R, m, T, carrier_freq, maxN)

	return t, Efields_dopp_arr




# Doppler with random path



@njit
def doppler_wave(t, m, T, collision_rate, freq, phase):

	'''
	Generates a continuous wave that changes frequency according to collision times
	sampled from a Poisson ditribution (maxwell.random_velocities)

	The frequencies considered are Doppler shifted accoring to particle velocity

	Input

		t - numpy array with time values
		m - mass of the particle (for random_velocities)
		T - temperature (for random_velocities)
		collision_rate - collision rate (for random_velocities)
		freq - frequency of the laser (not shifted)
		phase - initial phase of the wave

	Output

		wave - numpy aray with the wave function for all times t

	'''

	v_arr, t_ind = mx.random_velocities(t, m, T, collision_rate)

	freq_arr = freq*np.abs(v_arr)/ct.c

	phi = phase
	wave = np.zeros(shape=len(t), dtype=np.csingle)
	wave[t_ind[0]:t_ind[1]] = np.exp(1j*freq_arr[t_ind[0]]*t[t_ind[0]:t_ind[1]] + 1j*phi)

	for i in range(1, len(t_ind)-1):

		ind_0 = t_ind[i]
		ind_1 = t_ind[i+1]

		prev_freq = freq_arr[ind_0-1]
		new_freq = freq_arr[ind_0]

		phi = (prev_freq-new_freq)*t[ind_0] + phi

		wave[ind_0:ind_1] = np.exp(1j*new_freq*t[ind_0:ind_1] + 1j*phi)


	return wave



@njit
def cavityField_doppPath(tau, t0, D, R, m, T, collision_rate, carrier_freq, minAmp):

	'''
	Simulates the Doppler-shifted electric field inside the laser cavity

	Input

		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)	
		m - mass of the particle
		T - temperature in K
		collision_rate - collision rate
		carrier_freq - frequency of the laser

	Return 

		field - numpy array with the complex valued Doppler shifted cavity field amplitude
		t - numpy array with the time values (from t_maxn function)
		maxN - max reflection calculated (from t_maxn function)

	'''

	t, maxN = t_maxn_dopp(tau, t0, D, R, minAmp)

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		wave_dopp = doppler_wave(t, m, T, collision_rate, carrier_freq, phi[i])
		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*wave_dopp

	field = normConst_gaussSq(tau)*field

	return field, t, maxN



@njit
def cavityField_doppPath_rawCycle(t, tau, t0, D, R, m, T, collision_rate, carrier_freq, maxN):

	'''
	Similar function to cavityField_doppPath but takes as input t and maxN

	Input

		t - numpy array with the time values
		tau - duration of the initial Gaussian pulse
		t0 - center of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		minAmp - minimum amplitude to consider (between 0.0 and 1.0)	
		m - mass of the particle
		T - temperature
		collision_rate - collision rate
		carrier_freq - frequency of the laser
		maxN - maximum relfection to consider

	Return 

		field - numpy array with the complex valued Doppler shifted cavity field amplitude

	'''

	phi = np.random.uniform(0.0, 2*np.pi, size=maxN)

	field = np.zeros(shape=len(t), dtype=np.csingle)

	for i in range(maxN):

		wave_dopp = doppler_wave(t, m, T, collision_rate, carrier_freq, phi[i])
		field += (R**i)*gauss(t, t0+i*(D/ct.c), tau)*wave_dopp

	field = normConst_gaussSq(tau)*field

	return field



@njit(parallel=True)
def simul_fields_doppPath(nsim, tau, D, R, m, T, collision_rate, carrier_freq, minAmp=0.01):

	'''
	Generates several Doppler shifted cavity fields in a cycle (parallelized with Numba)
	Velocities acoording to a random path with Poisson sampled collisions

	Input

		nsim - number of fields to simulate
		tau - duration of the initial Gaussian pulse
		D - cavity diameter
		R - cavity reflectance
		m - mass of the particle
		T - temperature
		collision_rate - collision rate
		carrier_freq - frequency of the laser

	Return

		t - numpy array with the time values used
		Efields_arr - numpy array with all the simulated fields

	'''

	Efield_dopp, t, maxN = cavityField_doppPath(tau, 4*tau, D, R, 0.01, m, T, collision_rate, carrier_freq)

	Efields_dopp_arr = np.zeros(shape=(nsim, len(t)), dtype=np.csingle)

	Efields_dopp_arr[0,:] = Efield_dopp

	for i in prange(1, nsim):

		Efields_dopp_arr[i,:] = cavityField_doppPath_rawCycle(t, tau, 4*tau, D, R, m, T, collision_rate, carrier_freq, maxN)

	return t, Efields_dopp_arr

