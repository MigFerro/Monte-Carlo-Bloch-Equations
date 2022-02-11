#!/usr/bin/env python

'''
A list of:
	-Physical constants
	-Important constant values (masses, matrix elements, etc.)
	-Conversions
'''

e = 1.60217662E-19 # electron charge (C)
eps0 = 8.8541878128E-12 # vaccuum permittivity (F/m)

kB = 1.3806E-23 # boltzmann constant (J/K)
kB_eV = kB/e # boltzmann constant (eV/K)

alpha =  0.00729735257 #fine structure constant (adim)

u = 1.66054E-27 # 1 Dalton (kg)
m_mu = 0.1134289*u # muon mass (kg)
m_p = 1.00726467*u # proton mass (kg)
m_muH = m_mu + m_p #muH mass - no binding energy (kg)
m_H2 = 1.00784*u # H2 mass (kg)

c = 299792458 # speed of light (m/s)
hbar = 1.054571800E-34 #reduced Planck constant (SI)

LHD = 4.25E22 # liquid hydrogen density (at/cm^3)

MM1 = 1.228E-15 # magnetic dipole matrix element for muH HFS (m)

#conversions

bar2Pa = 1.0E5 # bar to Pa conversion
