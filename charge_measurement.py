import numpy as np
import matplotlib.pyplot as plt


def measure_charge():
    # _m is in metres, _px is in pixels

    # ----------------
    # system properties
    g = 9.81
    length_m = 0.9  # length of hanging string
    mass = 0.3 * 1e-3

    # ----------------
    # electric field
    voltage = 10 * 1e3
    plate_separation_m = 1e-1  # distances in cm converted to m
    electric_field = voltage / plate_separation_m

    # ----------------
    # lengths
    diameter_m = 8 * 1e-3
    diameter_px = 915 - 679  # 236 - as long as the camera doesn't move it should stay as 236
    dx_px = np.array([675 - 49])  # change in pixel value for desired measurement
    conversion = diameter_m / diameter_px  # metres per pixel
    # dx_m = dx_px * conversion  # change in position in metres
    dx_m = 1e-6  # change in position in metres

    # ----------------
    # change in charge
    dq = dx_m * ((mass * g) / (electric_field * length_m))  # change in charge in Coulombs
    print(f"{dq = }")


def plot_hysteresis():
    plate_separation = 0.1  # todo check
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
