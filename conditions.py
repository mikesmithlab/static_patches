import numpy as np


def get_conditions(filename=None):
    # if filename is given, conds comes from file (or is written to it if it doesn't exist)
    # if filename isn't given, conds defaults to the below dictionary
    conds = {
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
        'pos': np.array([0.001, -0.001, 0]),
        'velocity': np.array([0.1 * 2 ** 0.4, 0.1 * 2 ** 0.6, 0]),
        'omega': 0 * np.array([50 * 2 * np.pi, 5 * 2 * np.pi, 12 * 2 * np.pi]),
        'time_end': 6,
        'container_time_end': 5.8,
        'time_warp': 2 / 20,
        'refresh_rate': 60,
    }

    # ---------------------------------------
    # Don't change anything below this point!
    # ---------------------------------------

    if filename is not None:
        try:
            file = open(filename, "r")
            try:
                file.readline()
                field = file.readline().strip().split(",")
                conds = {
                    'g': float(field[0]),
                    'radius': float(field[1]),
                    'density': float(field[2]),
                    'coefficient_of_restitution': float(field[3]),
                    'mu': float(field[4]),
                    'gamma_t': float(field[5]),
                    'container_radius': float(field[6]),
                    'container_amplitude': float(field[7]),
                    'container_omega': float(field[8]),
                    'number_of_patches': int(field[9]),
                    'optimal_offset': int(field[10]),
                    'pos': np.array([float(field[11]), float(field[12]), float(field[13])]),
                    'velocity': np.array([float(field[14]), float(field[15]), float(field[16])]),
                    'omega': np.array([float(field[17]), float(field[18]), float(field[19])]),
                    'time_end': float(field[20]),
                    'container_time_end': float(field[21]),
                    'time_warp': float(field[22]),
                    'refresh_rate': float(field[23]),
                }
                print(f"Read properties and initial conditions from {filename}.")
            except ValueError:
                print(f"Wrong format of given file: {filename}. Ignoring the file - using default conds.")
        except FileNotFoundError:
            print(f"Can't find {filename}. Making new with default conds.")
            file = open(filename, "w")
            l1 = (
                f"g,radius,density,coefficient_of_restitution,mu,gamma_t,container_radius,container_amplitude,"
                f"container_omega,container_stop_time,number_of_patches,optimal_offset,pos(3),velocity(3),omega(3),"
                f"time_end,time_warp,refresh_rate\n"
            )
            p = conds['pos']
            v = conds['velocity']
            o = conds['omega']
            l2 = (
                f"{conds['g']},{conds['radius']},{conds['density']},{conds['coefficient_of_restitution']},"
                f"{conds['mu']},{conds['gamma_t']},{conds['container_radius']},{conds['container_amplitude']},"
                f"{conds['container_omega']},{conds['number_of_patches']},{conds['optimal_offset']},"
                f"{p[0]},{p[1]},{p[2]},{v[0]},{v[1]},{v[2]},{o[0]},{o[1]},{o[2]},"
                f"{conds['time_end']},{conds['container_time_end']},{conds['time_warp']},{conds['refresh_rate']}"
            )
            file.writelines(l1)
            file.writelines(l2)
        file.close()

    # object properties
    conds['mass'] = conds['density'] * (4 / 3) * np.pi * conds['radius'] ** 3
    conds['moment_of_inertia'] = (2 / 5) * conds['mass'] * conds['radius'] ** 2
    conds['spring_constant'] = (100 * conds['mass'] / conds['radius']) * (
            (conds['g'] ** 2) ** 0.5 + 2 * conds['container_amplitude'] * conds['container_omega'] ** 2)
    conds['damping'] = (-2 / np.pi) * conds['mass'] * np.log(conds['coefficient_of_restitution']) * (
            conds['spring_constant'] / conds['mass']) ** 0.5

    # timing
    conds['time_step'] = (1 / 50) * np.pi * np.sqrt((conds['mass'] / conds['spring_constant']))
    conds['total_steps'] = int(conds['time_end'] / conds['time_step'])
    # total_steps = (t_end / t_step) + ((int(t_end / t_step) - (t_end / t_step))) != 0)  # todo round up integer?
    conds['store_interval'] = int(conds['time_warp'] / (conds['time_step'] * conds['refresh_rate']))
    conds['total_store'] = int(conds['total_steps'] / conds['store_interval']) + (
            conds['total_steps'] % conds['store_interval'] > 0)  # round up int because of 0th frame

    return conds
