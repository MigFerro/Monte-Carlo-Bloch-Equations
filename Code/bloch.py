#!/usr/bin/env python

'''
Functions used for solving the Bloch equations numerically
with Runge-Kutta method (4th order)

Most functions make use of the Numba library to improve performance

'''

# standard library
import numpy as np
from numba import njit, prange

# local source library
import constants as ct


@njit
def bloch_2lvl_noGammas(t, rho, rabi, detune):

	'''
	Defines the Bloch equations for a 2-level system with no decay rates (no spont. emission or any collisions)

	Input

		t - time values
		rho - array with all populations anr coherences to calculate (rho11, rho12, rho22)
		rabi - rabi frequency
		detune - laser frequency detune

	Return

		array with Bloch equations (eq11, eq12, eq22, eq33)

	'''

	#rho = np.array([rho11, rho12, rho22, rho33])

	eq11 = -np.imag(rabi*rho[1]*np.exp(1j*detune*t))
	eq12 = (1j/2)*np.conjugate(rabi)*(rho[0]-rho[2])*np.exp(-1j*detune*t)
	eq22 = -eq11

	rho_dot = np.array([eq11, eq12, eq22])

	return rho_dot



@njit
def solveBlochRK4_2lvl(t, rabi, detune, bound):

	'''
	Solves the 2-level Bloch equations (no broadenings) numerically with Runge-Kutta (4th order) method

	Input

		t - time values
		rabi - rabi frequency
		detune - laser frequency detune
		bound - initial conditions for all populations and coherences

	Return

		numpy array with values of all populations and coherences for all times t

	'''

	nt = len(t)
	t0 = t[0]
	dt = t[1]-t0

	nx = len(bound)
	x = np.zeros(shape=(nx, nt), dtype=bound.dtype)
	x[:,0] = bound

	for i in range(nt-1):

		k1 = dt*bloch_2lvl_noGammas(t[i], x[:,i], rabi[i], detune)
		k2 = dt*bloch_2lvl_noGammas(t[i] + dt/2, x[:,i] + k1/2, rabi[i], detune)
		k3 = dt*bloch_2lvl_noGammas(t[i] + dt/2, x[:,i] + k2/2, rabi[i], detune)
		k4 = dt*bloch_2lvl_noGammas(t[i] + dt, x[:,i] + k3, rabi[i], detune)

		dx = (k1 + 2*k2 + 2*k3 + k4)/6

		x[:,i+1] = x[:,i] + dx

	return x



@njit(parallel = True)
def simul_2lvl(t, rabi, detune, bound, nsim):

	'''
	Solves the Bloch equations numerically with Runge-Kutta (4th order) method

	Solves the equations several times in a cycle for an array of different rabi frequencies
	to obtain an average population

	Input

		t - time values
		rabi - rabi frequency
		detune - laser frequency detune
		bound - initial conditions for all populations and coherences
		nsim - number of cycles (must be <= len(rabi))

	Return

		average populations and coherences for all times t

	'''

	rho = np.zeros(shape=(nsim, len(bound), len(t)), dtype=bound.dtype)

	for i in prange(nsim):
		rho[i] = solveBlochRK4_2lvl(t, rabi[i], detune, bound)

	return np.sum(rho, axis=0)/nsim



@njit
def bloch_3lvl(t, rho, rabi, params):

	'''
	Defines the Bloch equations for a 3-level system |1>, |2>, |3>
	where level |3> is only accessible via inelastic collisions from |2>

	Input

		t - time values
		rho - array with all populations anr coherences to calculate (rho11, rho12, rho22, rho33)
		rabi - rabi frequency
		params - paramaters (detune, spont. emission, el. collision, inel. collision)

	Return

		array with Bloch equations (eq11, eq12, eq22, eq33)

	'''

	detune = params[0]
	gamma_sp = params[1]
	gamma_el11 = params[2]
	gamma_el22 = params[3]
	gamma_inel = params[4]
	gamma_l = params[5]

	gamma_p = gamma_sp #population broadening
	gamma_c = gamma_sp + gamma_el11 + gamma_el22 + gamma_inel + gamma_l #coherence broadening

	#rho = np.array([rho11, rho12, rho22, rho33])

	eq11 = -np.imag(rabi*rho[1]*np.exp(1j*detune*t)) + gamma_p*rho[2]
	eq12 = (1j/2)*np.conjugate(rabi)*(rho[0]-rho[2])*np.exp(-1j*detune*t)-(gamma_c/2)*rho[1]
	eq22 = -eq11 - gamma_inel*rho[2]
	eq33 = gamma_inel*rho[2]

	rho_dot = np.array([eq11, eq12, eq22, eq33])

	return rho_dot



@njit
def solveBlochRK4_3lvl(t, rabi, params, bound):

	'''
	Solves the 3-level Bloch equations numerically with Runge-Kutta (4th order) method

	Input

		t - time values
		rabi - rabi frequency
		params - parameters (detune, spont. emission, el. collision, inel. collision, laser bandwidth)
		bound - initial conditions for all populations and coherences

	Return

		numpy array with values of all populations and coherences for all times t

	'''

	nt = len(t)
	t0 = t[0]
	dt = t[1]-t0

	nx = len(bound)
	x = np.zeros(shape=(nx, nt), dtype=bound.dtype)
	x[:,0] = bound

	for i in range(nt-1):

		k1 = dt*bloch_3lvl(t[i], x[:,i], rabi[i], params)
		k2 = dt*bloch_3lvl(t[i] + dt/2, x[:,i] + k1/2, rabi[i], params)
		k3 = dt*bloch_3lvl(t[i] + dt/2, x[:,i] + k2/2, rabi[i], params)
		k4 = dt*bloch_3lvl(t[i] + dt, x[:,i] + k3, rabi[i], params)

		dx = (k1 + 2*k2 + 2*k3 + k4)/6

		x[:,i+1] = x[:,i] + dx

	return x



@njit(parallel = True)
def solveAverage_field_3lvl(t, rabi, params, bound, nsim):

	'''
	Solves the Bloch equations numerically with Runge-Kutta (4th order) method

	Solves the equations several times in a cycle for an array of different rabi frequencies
	to obtain an average population

	Input

		t - time values
		rabi - rabi frequency
		params - parameters (detune, spont. emission, el. collision, inel. collision)
		bound - initial conditions for all populations and coherences
		nsim - number of cycles (must be <= len(rabi))

	Return

		average populations and coherences for all times t

	'''

	rho = np.zeros(shape=(nsim, len(bound), len(t)), dtype=bound.dtype)

	for i in prange(nsim):
		rho[i] = solveBlochRK4_3lvl(t, rabi[i], params, bound)

	return np.sum(rho, axis=0)/nsim



@njit
def rabiFreq(field, F, M):

	'''
	Calculates the rabi frequency for a given complex field amplitude, fluence and matrix element

	Input

		field - complex valued field amplitude
		F - laser fluence
		M - transition matrix element

	Return

		rabi frequency

	'''

	return M*(ct.e/ct.hbar)*np.sqrt(2*F/(ct.eps0*ct.c))*field
