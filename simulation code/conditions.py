import numpy as np
import json
from my_tools import offset_finder

def params():
    return {
        'g': -9.81,
        'radius': 5 / 1000,
        'density': 1000,
        'coefficient_of_restitution': 0.8,
        'mu': 0.5,
        'gamma_t': 0.8,
        'container_radius': 20 / 1000,
        'container_amplitude': 2 / 1000,
        'container_omega': 20 * 2 * np.pi,
        'number_of_patches': int(200),
        'optimal_offset': 0.4383841477800122,  # if number_of_patches is not 200, this needs changing as well
        'pos': [0.001, -0.001, 0],#np.array([0.001, -0.001, 0]),
        'velocity': [0.1 * 2 ** 0.4, 0.1 * 2 ** 0.6, 0],#np.array([0.1 * 2 ** 0.4, 0.1 * 2 ** 0.6, 0]),
        'omega': 0*[50 * 2 * np.pi, 5 * 2 * np.pi, 12 * 2 * np.pi],# 0 * np.array([50 * 2 * np.pi, 5 * 2 * np.pi, 12 * 2 * np.pi]),
        'time_end': 6,
        'container_time_end': 5.8,
        'time_warp': 2 / 20,
        'refresh_rate': 60,
    }


def sim_params(save_filename='test.json'):
    #Call user params
    conds = params()

    # calc object properties
    conds['mass'] = conds['density'] * (4 / 3) * np.pi * conds['radius'] ** 3
    conds['moment_of_inertia'] = (2 / 5) * conds['mass'] * conds['radius'] ** 2
    conds['spring_constant'] = (100 * conds['mass'] / conds['radius']) * (
            (conds['g'] ** 2) ** 0.5 + 2 * conds['container_amplitude'] * conds['container_omega'] ** 2)
    conds['damping'] = (-2 / np.pi) * conds['mass'] * np.log(conds['coefficient_of_restitution']) * (
            conds['spring_constant'] / conds['mass']) ** 0.5

    # simulation time settings
    conds['time_step'] = (1 / 50) * np.pi * np.sqrt((conds['mass'] / conds['spring_constant']))
    conds['total_steps'] = int(conds['time_end'] / conds['time_step'])
    conds['store_interval'] = int(conds['time_warp'] / (conds['time_step'] * conds['refresh_rate']))
    conds['total_store'] = int(conds['total_steps'] / conds['store_interval']) + (
                conds['total_steps'] % conds['store_interval'] > 0)  # round up int because of 0th frame

    # find optimal offset for any number of patches
    conds['optimal_offset'] = offset_finder(conds['number_of_patches'])

    with open(save_filename, 'w') as simulation_conditions:
        simulation_conditions.write(json.dumps(conds))

    return conds
