#!/usr/bin/env python

'''
Functions used for reading/writing files generated such as:
	- collision rates
	- time arrays
	- cavity fields
	- populations obtained from the Bloch equations

'''

# standard library
import numpy as np
import os

# local source library
import constants as ct
import maxwell as mx


def format_time(t):

	'''
	Function for converting a time value (e.g. from time.perf_counter()) to
	a formatted string displaying hours, minutes and seconds

	Input

		t - time value

	Return

		formatted string

	'''

	if t >= 3600.0:
		return f"{round(t/3600, 1)} hours"
	elif t >= 60.0:
		return f"{round(t/60, 2)} minutes"
	else:
		return f"{round(t, 2)} seconds"


def write_collision_rates(path, T, stat=False):
	
	'''
	Function for writing average collision rates to a .dat file

	Reads the collision rates from a .dat file which are given in fuction of muH energy
	and averages the rates based on a Maxwell-Boltzmann energy distribution for a given temperature

	Input

		path - path of the files to read

		T - temperature for MB distribution average

		stat - if the rates are obtained for a statistical distribution of ortho (75%) and para-hydrogen (25%) (True)
				or for a Boltzmann distribution (False)

	'''

	if stat:
		file = os.path.join(path, r'rates_'+str(int(T))+'K_stat_raw.dat')
	else:
		file = os.path.join(path, r'rates_'+str(int(T))+'K_raw.dat')

	data = np.loadtxt(file)

	muHenergies = data[:,0]
	rates = data[:,1::]

	dE = muHenergies[1]-muHenergies[0]

	prob = mx.MB_energy_dist(T, muHenergies)*dE

	res = np.zeros(shape=3)
	for i in range(len(res)):
		res[i] = np.sum(rates[:,i]*prob)

	if stat:
		with open(os.path.join(os.path.join(path,'Average'), r'average_rates_'+str(int(T))+'K_stat.dat'), 'w') as f:
			np.savetxt(f, res)

	else:
		with open(os.path.join(os.path.join(path,'Average'), r'average_rates_'+str(int(T))+'K.dat'), 'w') as f:
			np.savetxt(f, res)


def read_average_collision_rates(path, T, stat=False):
	
	'''
	Function for reading average collision rates from a .dat file

	Rates are given for liquid hydrogen density (LHD)

	Input

		path - path of the file to read

		T - temperature for which the rates where calculated

		stat - if the rates are obtained for a statistical distribution of ortho (75%) and para-hydrogen (25%) (True)
				or for a Boltzmann distribution (False)
	
	Return
		
		rates - array [elastic (F=0), elastic (F=1), inelastic (F=1->F=0)]

	'''

	if stat:
		file = os.path.join(path, r'average_rates_'+str(int(T))+'K_stat.dat')
	else:
		file = os.path.join(path, r'average_rates_'+str(int(T))+'K.dat')

	rates = np.loadtxt(file)

	return rates


def write_time(path, t, tau, D, R):
	
	'''
	Function for writing a time array of a cavity field to an .out file 

	Input

		path - path of the file to write on

		t - numpy array with time values

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance
	
	The file is named as

		t_tau_D_R.out

	where tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		t_10_10_995.out

	'''

	name_t = f't_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_t), 'w') as f:
		np.savetxt(f, t)


def read_time(path, tau, D, R):
	
	'''
	Function for reading a time array of a cavity field from an .out file 

	Input

		path - path of the file to write on

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance
	
	Return

		t - numpy array with time values

	The file should be named

		t_tau_D_R.out

	where tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		t_10_10_995.out
	
	'''	

	name_t = f't_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_t), 'r') as f:
		t = np.loadtxt(f)

	return t


def write_time_dopp(path, t, T, P, tau, D, R):
	
	'''
	Function for writing a time array of a doppler-shifted cavity field to an .out file 

	Input

		path - path of the file to write on

		t - numpy array with time values

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance
	
	The file is named as

		t_dopp_T_P_tau_D_R.out

	where T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		t_dopp_50_5_10_10_995.out

	'''

	name_t = f't_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_t), 'w') as f:
		np.savetxt(f, t)


def read_time_dopp(path, T, P, tau, D, R):
	
	'''
	Function for reading a time array of a doppler-shifted cavity field from an .out file 

	Input

		path - path of the file to write on

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance
	
	Return

		t - numpy array with time values

	The file should be named

		t_dopp_T_P_tau_D_R.out

	where T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		t_dopp_50_5_10_10_995.out

	'''

	name_t = f't_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_t), 'r') as f:
		t = np.loadtxt(f)

	return t


def write_fields(path, fields, tau, D, R):
	
	'''
	Function for writing an array of cavity fields to an .out file 

	Input

		path - path of the file to write on

		fields - numpy array with complex valued cavity field amplitudes

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance
	
	The file is named as

		fields_tau_D_R.out

	where tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_10_10_995.out

	'''

	name_f = f'fields_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, fields)


def read_fields(path, tau, D, R):
	
	'''
	Function for reading an array of cavity fields to an .out file 

	Input

		path - path of the file to write on

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance
	
	Return

		Efield_arr - numpy array with complex valued cavity field amplitudes

	The file should be named

		fields_tau_D_R.out

	where tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_10_10_995.out
	
	'''

	name_f = f'fields_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		Efield_arr_str = np.loadtxt(f, dtype=str)
		Efield_arr = np.zeros(shape=Efield_arr_str.shape, dtype=np.csingle)
		for i in range(len(Efield_arr)):
			Efield_arr[i] = np.array([np.complex(x) for x in Efield_arr_str[i]]) # this workaround is needed to read the complex values from file

	return Efield_arr


def write_fields_dopp(path, fields, T, P, tau, D, R):
	
	'''
	Function for writing an array of doppler-shifted cavity fields to an .out file 

	Input

		path - path of the file to write on

		fields - numpy array with complex valued doppler-shifted cavity field amplitudes

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance

	The file is named as

		fields_dopp_T_P_tau_D_R.out

	where T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_dopp_50_5_10_10_995.out

	'''

	name_f = f'fields_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, fields)


def read_fields_dopp(path, T, P, tau, D, R):
	
	'''
	Function for reading an array of doppler-shifted cavity fields from an .out file 

	Input

		path - path of the file to write on

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity field

		D - cavity diameter

		R - cavity reflectance
	
	Return

		Efield_arr - numpy array with complex valued doppler-shifted cavity field amplitudes

	The file should be named

		fields_dopp_T_P_tau_D_R.out

	where T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_dopp_50_5_10_10_995.out

	'''

	name_f = f'fields_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		Efield_arr_str = np.loadtxt(f, dtype=str)
		Efield_arr = np.zeros(shape=Efield_arr_str.shape, dtype=np.csingle)
		for i in range(len(Efield_arr)):
			Efield_arr[i] = np.array([np.complex(x) for x in Efield_arr_str[i]])


	return Efield_arr


def write_r33_avg(path, r33, F, tau, D, R):
	
	'''
	Function for writing the average population obtained for an array of cavity fields to an .out file 

	Input

		path - path of the file to write on

		r33 - numpy array with the population values in time
	
		F - laser fluence 

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance
	
	The file is named as

		r33_F_tau_D_R.out

	where F is in J/m^2, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		F = 1000.0 J/m^2 = 1.0 J/cm^2 
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		r33_1000_10_10_995.out

	'''


	name_r = f'r33_{int(F)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'w') as f:
		np.savetxt(f, r33)


def read_r33_avg(path, F, tau, D, R):
	
	'''
	Function for reading the average population obtained for an array of cavity fields from an .out file 

	Input

		path - path of the file to read from
	
		F - laser fluence 

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance
	
	Return

		r33 - numpy array with the population values in time

	The file should be named

		r33_F_tau_D_R.out

	where F is in J/m^2, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		F = 1000.0 J/m^2 = 1.0 J/cm^2 
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		r33_1000_10_10_995.out

	'''

	name_r = f'r33_{int(F)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'r') as f:
		r33 = np.loadtxt(f)

	return r33


def write_r33_avg_dopp(path, r33, F, T, P, tau, D, R):
	
	'''
	Function for writing the average population obtained for an array of doppler-shifted cavity fields to an .out file 

	Input

		path - path of the file to write on

		r33 - numpy array with the population values in time

		F - laser fluence 

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance
	
	The file is named as

		r33_dopp_F_T_P_tau_D_R.out

	where F is in J/m^2, T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		F = 1000.0 J/m^2 = 1.0 J/cm^2
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_dopp_1000_50_5_10_10_995.out

	'''

	name_r = f'r33_dopp_{int(F)}_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'w') as f:
		np.savetxt(f, r33)


def read_r33_avg_dopp(path, F, T, P, tau, D, R):
	
	'''
	Function for reading the average population obtained for an array of doppler-shifted cavity fields from an .out file 

	Input

		path - path of the file to read from

		F - laser fluence 

		T - temperature for which the field was calculated

		P - temperature for which the field was calculated

		tau - pulse duration of the cavity fields

		D - cavity diameter

		R - cavity reflectance

	Return

		r33 - numpy array with the population values in time
	
	The file should be named

		r33_dopp_F_T_P_tau_D_R.out

	where F is in J/m^2, T is in K, P is in bar/10, tau is in ns, D in cm and R is a value between 0 and 1000
	E.g.
		F = 1000.0 J/m^2 = 1.0 J/cm^2
		T = 50.0 K
		P = 0.5 bar
		tau = 10.0 ns
		D = 10.0 cm
		R = 0.995

		fields_dopp_1000_50_5_10_10_995.out

	'''

	name_r = f'r33_dopp_{int(F)}_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'r') as f:
		r33 = np.loadtxt(f)

	return r33



# Functions used for testing only


def write_field_avg(path, field, tau, D , R):

	name_f = f'field_avg_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, field)


def read_field_avg(path, tau, D , R):

	name_f = f'field_avg_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		field_avg = np.loadtxt(f)

	return field_avg


def write_field_avg_dopp(path, field, T, P, tau, D , R):

	name_f = f'field_avg_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, field)


def read_field_avg_dopp(path, T, P, tau, D , R):

	name_f = f'field_avg_dopp_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		field_avg = np.loadtxt(f)

	return field_avg


def write_r33_avg_field(path, r33, F, tau, D, R):

	name_r = f'r33_avg_field_{int(F)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'w') as f:
		np.savetxt(f, r33)	


def read_r33_avg_field(path, F, tau, D, R):

	name_r = f'r33_avg_field_{int(F)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'r') as f:
		r33 = np.loadtxt(f)

	return r33


def write_r33_avg_field_dopp(path, r33, F, T, P, tau, D, R):

	name_r = f'r33_avg_field_dopp_{int(F)}_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'w') as f:
		np.savetxt(f, r33)	


def read_r33_avg_field_dopp(path, F, T, P, tau, D, R):

	name_r = f'r33_avg_field_dopp_{int(F)}_{int(T)}_{int(P*10/ct.bar2Pa)}_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}.out'

	with open(os.path.join(path, name_r), 'r') as f:
		r33 = np.loadtxt(f)

	return r33


def write_fields_gauss(path, field, tau, D, R, tau_fact):

	name_f = f'field_g_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}_{int(tau_fact)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, field)


def read_field_gauss(path, tau, D, R, tau_fact):

	name_f = f'field_g_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}_{int(tau_fact)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		field_gauss = np.loadtxt(f)

	return field_gauss


def write_fields_rect(path, field, tau, D, R, tau_fact):

	name_f = f'field_r_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}_{int(tau_fact)}.out'

	with open(os.path.join(path, name_f), 'w') as f:
		np.savetxt(f, field)


def read_field_rect(path, tau, D, R, tau_fact):

	name_f = f'field_r_{int(tau*1E9)}_{int(D*1E2)}_{int(R*1000)}_{int(tau_fact)}.out'

	with open(os.path.join(path, name_f), 'r') as f:
		field_rect = np.loadtxt(f)

	return field_rect


