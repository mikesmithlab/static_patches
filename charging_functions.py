import numpy as np


def charge_decay_function(charge, time_interval):  # returns the new charges due to decay (to air?)
    # todo
    # does the charge spread to nearby patches? <-- would be horrible to compute

    # decay_constant = -0.01  # approximate decay constant from experimental data. Half life is approx 11.5 mins
    # decay_to_charge = 0  # 0.38 * 1e-9 / n  # for particle, unknown for container! (from experimental data)
    # return (charge - decay_to_charge).dot(np.exp(decay_constant * time_interval)) + decay_to_charge
    return charge.dot(np.exp(-0.01 * time_interval))


def charge_hit_function(patch_charge_part, patch_charge_cont):  # returns the new charges of colliding patches
    # todo:
    # do previous charges of the patches matter? or just add some constant every collision? ('proper "saturation"'?)
    # does the force matter?
    # does the charge of nearby patches matter?
    # constant that is added needs to change with patch area (work area out once then input it to this function)

    # charge_per_hit = 4e-13  # this number needs changing
    # return patch_charge_part + charge_per_hit, patch_charge_cont - charge_per_hit
    return patch_charge_part + 4e-13, patch_charge_cont - 4e-13
