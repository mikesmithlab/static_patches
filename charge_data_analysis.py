import numpy as np
import matplotlib.pyplot as plt


bead1_times = np.cumsum(np.array([0, 1, 5, 5, 5]))
bead1_charges = np.array([-0.116, 0.008, 0.277, 0.436, 0.311])

bead2_times = np.cumsum(np.array([0, 1, 1, 1, 1, 1]))
bead2_charges = np.array([-1.575, -0.554, -0.081, 0.268, 0.546, 0.743])

bead3_times = np.cumsum(np.array([0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]))
bead3_charges = np.array([-0.469, 0.493, 0.909, 1.267, 1.323, 1.478, 1.506, 1.609, 1.571, 1.665, 1.546, 1.539, 1.232])

fig_c = plt.figure()
fig_c.canvas.manager.set_window_title("Charge against time")
plt.plot(bead1_times, bead1_charges, '.', bead2_times, bead2_charges, '.', bead3_times, bead3_charges, '.', )
plt.xlabel("Time/s")
plt.ylabel("Charge/nC")
plt.show()
