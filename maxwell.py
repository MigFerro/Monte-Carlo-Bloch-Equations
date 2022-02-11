#!/usr/bin/env python

'''
Functions based on Maxwell-Boltzmann statistics
	- MB energy distribution (T, E)
	- MB velocity generator (m, T)
	- random path generator
	- random path velocities

All functions make use of the Numba library to improve performance

'''

# standard library
import numpy as np

# Numba
from numba import njit, vectorize

# local source library
import constants as ct


@njit
def idealGasDensity(T, P):

	'''
	Calculates the ideal gas density (in atoms/cm^3) for a given temperature and pressure

	Input

		T - temperature in K
		P - pressure in Pa

	Return

		ideal gas density in atoms/cm^3

	'''

	#1.0E-6 factor for value in atoms/cm^3

	return P/(ct.kB*T)*1.0E-6



@njit
def MB_energy_dist(T, E):

	'''
	Maxwell-Boltzmann energy distribution value for given temperature and energy

	Input

		T - temperature in K
		E - energy in eV

	Return

		value of the MB energy distribution (float)

	'''

	return 2*(np.sqrt(1/np.pi))*((ct.kB_eV*T)**(-3/2))*(np.sqrt(E))*np.exp(-E/(ct.kB_eV*T))



@njit
def MB_velocity_dist(m, T, v):

	'''
	Maxwell-Boltzmann velocity distribution value for given mass, temperature and velocity

	Input

		m - mass in kg
		T - temperature in K
		v - velocity in m/s

	Return

		value of the MB velocity distribution (float)

	'''

	return (np.sqrt(m/(2*np.pi*ct.kB*T)))*np.exp(-(m*v**2)/(2*ct.kB*T))



@njit
def MB_velocity(m, T, size=None):

	'''
	Maxwell-Boltzmann velocity generator
	Samples an array of velocities from a MB distribution for given mass and temperature

	Input

		m - mass of the particles in kg
		T - temperature in K
		size - size of the array (number of velocities to generate)

	Return

		numpy array with the generated velocities

	'''

	sigma = np.sqrt(ct.kB*T/m)

	return np.random.normal(loc=0, scale=sigma, size=size)



@njit
def random_path_2D(t, m, T, collision_rate):

	'''
	Generates a 2D random path based on MB velocities and Poisson sampled collisions

	Input

		t - time array to calculate the path for
		m - mass of the particles in kg
		T - temperature in K
		collision_rate - collision rate of the particles at given temperature

	Return

		pos_arr - numpy array with x and y values for all values of t

	'''

	dt = t[1]-t[0]
	pos_arr = np.zeros(shape=(len(t),2))

	if dt >= 1/collision_rate:

		return pos_arr

	v = MB_velocity(m, T, size=2)
	p = dt*collision_rate
	c=0

	for i in range(len(t)):
		n = np.random.random()
		if n<p:
			v = MB_velocity(m, T, size=2)

		pos_arr[i] = pos_arr[i-1] + v*dt

	return pos_arr



@njit
def random_velocities(t, m, T, collision_rate):

	'''
	Generates MB velocities for Poisson sampled collisions

	Input

		t - time array to calculate the velocties for
		m - mass of the particles in kg
		T - temperature in K
		collision_rate - collision rate of the particles at given temperature

	Return

		v_arr - numpy array with v values for all values of t
		t_ind - indicies of t where a collision happened 

	'''

	dt = t[1]-t[0]
	v_arr = np.zeros(shape=len(t)) #empty v(t)
	t_ind = [0] #array with the time indices where collisions happened (also includes the indices 0 and len(t))

	v = MB_velocity(m, T, size=1)[0]
	p = dt*collision_rate

	v_arr[0] = v # first 2 instances of velocity
	v_arr[1] = v

	for i in range(2, len(t)-2):

		n = np.random.random()

		if n<p and i>t_ind[-1]+1: #imediate successive collisions are not allowed 
			v = MB_velocity(m, T, size=1)[0]
			t_ind.append(i)

		v_arr[i] = v

	v_arr[-2] = v
	v_arr[-1] = v

	t_ind.append(len(t))

	return v_arr, t_ind

