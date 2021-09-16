import numpy as np
import matplotlib.pyplot as plt


def measure_charge():
    # _m is in metres, _px is in pixels

    # ----------------
    # misc. system properties
    g = 9.81
    length_m = 0.95  # length of hanging string
    mass = 0.7 * 1e-3
    string_density = 0.0721e-3 / 2.56  # measured 2.56m of string (maybe a couple of cm more) and it was 0.0721g

    # ----------------
    # electric field
    voltage = 1.25 * 1e3
    plate_separation_m = 1e-1
    electric_field = voltage / plate_separation_m

    # ----------------
    # lengths
    diameter_m = 10 * 1e-3
    diameter_px = 915 - 679
    # dx_px = 675 - 49  # change in pixel value for desired measurement
    # conversion = diameter_m / diameter_px  # metres per pixel
    # dx_m = dx_px * conversion  # change in position in metres
    dx_m = diameter_m  # change in position in metres

    # ----------------
    # change in charge
    dq = dx_m * (mass + string_density * length_m / 2) * g / (electric_field * (length_m ** 2 - dx_m ** 2) ** 0.5)
    dq = dq * 10 ** 9  # convert from C to nC so we always have the same units
    # dq_bad = (dx_m * mass * g) / (electric_field * length_m)
    # change in charge in Coulombs
    print(f"{dq = } nC")


def plot_hysteresis():
    plate_separation = 0.1
    start = 0
    end = 5
    step = 0.25
    voltages = np.arange(start, end, step=step)
    voltages = np.array([*voltages, end, *voltages[::-1]]) * 1e3
    # print(f"{voltages = }")
    fields = voltages / plate_separation
    distances = np.linspace(0, 1, int(np.shape(voltages)[0])) * 1e-3
    plt.plot(fields, distances)
    plt.xlabel("field / V/m")
    plt.ylabel("distance / m")
    plt.show()


# --------------------------------
measure_charge()
# plot_hysteresis()
