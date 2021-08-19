import numpy as np


def get_options():
    opts = {
        'g': -9.81,
        'radius': 5 / 1000,
        'container_radius': 20 / 1000,
        'density': 1000,
        'container_amplitude': 2 / 1000,
        'container_frequency': 20,
        'coefficient_of_restitution': 0.8,
        'mu': 0.5,
        'gamma_t': 0.8,
        'time_end': 50,
        'number_of_patches': int(200),
        'pos': np.array([0.001, 0.002 ** 0.5, 0]),
        'velocity': np.array([0.02 ** 0.4, 0.02 ** 0.6, 0]),
        'omega': 0 * np.array([50 * 2 * np.pi, 5 * 2 * np.pi, 12 * 2 * np.pi]),
    }
    # exec(open(filename).read())

    # ---------------------------------------
    # Don't change anything below this point!
    # ---------------------------------------

    # object properties
    opts['mass'] = opts['density'] * (4 / 3) * np.pi * opts['radius'] ** 3
    opts['moment_of_inertia'] = (2 / 5) * opts['mass'] * opts['radius'] ** 2
    opts['spring_constant'] = (100 * opts['mass'] / opts['radius']) * (
            np.sqrt(np.sum(opts['g'] ** 2)) + 2 * opts['container_amplitude'] * opts['container_frequency'] ** 2)
    opts['damping'] = (-2) * opts['mass'] * np.sqrt(
        opts['spring_constant'] / opts['mass']) * np.log(opts['coefficient_of_restitution']) / np.pi

    # timing
    opts['time_step'] = (1 / 50) * np.pi * np.sqrt((opts['mass'] / opts['spring_constant']))
    opts['total_steps'] = int(opts['time_end'] / opts['time_step'])
    # total_steps = (t_end / t_step) + ((int(t_end / t_step) - (t_end / t_step))) != 0)  # todo round up integer?
    # opts['store_interval'] = int((6 / 50) / (opts['time_step'] * 60))  # 60Hz
    opts['store_interval'] = int((6 / 50) / (opts['time_step'] * 144))  # 144Hz
    opts['total_store'] = int(opts['total_steps'] / opts['store_interval']) + (
            opts['total_steps'] % opts['store_interval'] > 0)  # round up int because of 0th frame

    print(opts)
    return opts
