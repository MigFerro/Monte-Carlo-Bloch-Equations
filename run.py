import os

import constants as ct
import maxwell as mx
import field
import blochRK4 as bRK4
import fileIO as fIO

import numpy as np
from numba import njit, prange

import time

#directory of the scripts
path_drive = os.path.join('D:\\', 'Tese','CavitySim v2.0')
data_path = os.path.join(path_drive, r'data')
field_1k_path = os.path.join(data_path, r'field_1k')
field_avg_path = os.path.join(data_path, r'field_avg')
field_1k_dopp_old_path = os.path.join(data_path, r'field_1k_dopp_old')
field_1k_dopp_path = os.path.join(data_path, r'field_1k_dopp')
field_avg_dopp_path = os.path.join(data_path, r'field_avg_dopp')
pop_1k_path = os.path.join(data_path, r'pop_1k')
pop_1k_dopp_old_path = os.path.join(data_path, r'pop_1k_dopp_old')
pop_1k_dopp_path = os.path.join(data_path, r'pop_1k_dopp')
pop_avg_field_path = os.path.join(data_path, r'pop_avg_field')
pop_avg_field_dopp_path = os.path.join(data_path, r'pop_avg_field_dopp')
avg_rates_path = os.path.join(os.path.join(data_path, r'Rates'),'Average')
avg_pop_conv_path = os.path.join(data_path, r'pop_conv')
t_path = os.path.join(data_path, r'times')
t_dopp_path = os.path.join(data_path, r'times_dopp')

#program
if __name__ == '__main__':

	os.system('cls')

	hour_start = time.gmtime()

	t_start = time.perf_counter()
	t_sum_sim = 0.0

	print('\nSimulation started at: ' + time.strftime("%H:%M:%S", hour_start)+' (UTC+0) \n')

	t_sum_sim=0.0

	F_arr1 = np.array([0.5, 0.1]) * 1E4
	#F_arr1 = np.array([0.1, 0.2, 7.50, 10.0]) * 1E4
	#F_arr2 = np.array([0.2, 0.4, 15.0, 20.0]) * 1E4
	#F_arr3 = np.array([1.0, 2.0, 75.0, 100.0]) * 1E4

	T_arr = np.array([22.0, 50.0])
	P_arr = np.array([0.5])*ct.bar2Pa

	tau_arr = np.array([5.0, 10.0, 50.0, 100.0])*1E-9
	D_arr = np.array([1.0, 5.0, 20.0])*1E-2
	R_arr = np.array([0.995])

	nsim = 1000

	m = ct.m_muH
	freq = 2*np.pi*44.0E12
	M = ct.MM1
	detune = 0.0
	bound = np.array([1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j])

	# for numba compilation
	bRK4.solveAverage_field_3lvl(np.linspace(0,1,10), np.ones(shape=(2, 10)), np.array([0.0, 0.0, 0.0, 0.0]), bound, 2)

	total_sim = len(F_arr1)*len(T_arr)*len(P_arr)*len(tau_arr)*len(D_arr)*len(R_arr)
	#total_sim = len(T_arr)*len(P_arr)*len(tau_arr)*len(D_arr)*len(R_arr)
	curr_sim = 0

	for g in range(len(F_arr1)):
		for h in range(len(T_arr)):
			for i in range(len(P_arr)):

				dens = mx.idealGasDensity(T_arr[h], P_arr[i])
				rates = fIO.read_average_collision_rates(avg_rates_path, T_arr[h], stat=True)
				rates = rates*(dens/ct.LHD)
				rate = rates[0]+rates[1]
				params = np.array([0.0, 0.0, rate, rates[-1]])*2*np.pi #for rad/s

				for k in range(len(tau_arr)):
					for l in range(len(D_arr)):
						for j in range(len(R_arr)):
					
							curr_sim += 1

							print(f'Simulations per iteration: {nsim} \n')
							print('----------------------------------')
							print(f'\tIteration {curr_sim} of {total_sim}')
							print('----------------------------------\n')

							if t_sum_sim!=0:
								print('# ETA: '+fIO.format_time((t_sum_sim/(curr_sim-1))*(total_sim-curr_sim+1))+' #\n\n')
								print('Total running time: '+fIO.format_time(time.perf_counter()-t_start)+'\n')
							else:
								print('# info will be printed after 1st iteration #\n\n')

							t_start_sim = time.perf_counter()

							t = fIO.read_time_dopp(t_dopp_path, T_arr[h], P_arr[i], tau_arr[k], D_arr[l], R_arr[j])
							fields = fIO.read_fields_dopp(field_1k_dopp_path, T_arr[h], P_arr[i], tau_arr[k], D_arr[l], R_arr[j])
							
							rabi_1 = bRK4.rabiFreq(fields, F_arr1[g], M)
							#rabi_2 = bRK4.rabiFreq(fields, F_arr2[g], M)
							#rabi_3 = bRK4.rabiFreq(fields, F_arr3[g], M)

							r33_1 = bRK4.solveAverage_field_3lvl(t, rabi_1, params, bound, nsim)
							#r33_2 = bRK4.solveAverage_field_3lvl(t, rabi_2, params, bound, nsim)
							#r33_3 = bRK4.solveAverage_field_3lvl(t, rabi_3, params, bound, nsim)


							fIO.write_r33_avg_dopp(pop_1k_dopp_path, np.real(r33_1[3]), F_arr1[g], T_arr[h], P_arr[i], tau_arr[k], D_arr[l], R_arr[j])
							#fIO.write_r33_avg_dopp(pop_1k_dopp_path, np.real(r33_2[3]), F_arr2[g], T_arr[h], P_arr[i], tau_arr[k], D_arr[l], R_arr[g])
							#fIO.write_r33_avg_dopp(pop_1k_dopp_path, np.real(r33_3[3]), F_arr3[g], T_arr[h], P_arr[i], tau_arr[k], D_arr[l], R_arr[g])

							t_sim = time.perf_counter()-t_start_sim
							t_sum_sim += t_sim

							os.system('cls')

