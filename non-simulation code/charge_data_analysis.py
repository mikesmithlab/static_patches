import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 1st runs after cleaning ----------------------------------
# bead1_times = np.cumsum(np.a([0, 1, 5, 5, 5]) * 60)
# bead1_charges = np.a([-0.116, 0.008, 0.277, 0.436, 0.311])
#
# bead2_times = np.cumsum(np.a([0, 1, 1, 1, 1, 1]) * 60)
# bead2_charges = np.a([-1.575, -0.554, -0.081, 0.268, 0.546, 0.743])
#
# bead3_times = np.cumsum(np.a([0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]) * 60)
# bead3_charges = np.a([-0.469, 0.493, 0.909, 1.267, 1.323, 1.478, 1.506, 1.609, 1.571, 1.665, 1.546, 1.539, 1.232])
#
# fig_c1 = plt.figure()
# fig_c1.canvas.manager.set_window_title("Charge against time")
# plt.plot(bead1_times, bead1_charges, '.', bead2_times, bead2_charges, '.', bead3_times, bead3_charges, '.', )
# plt.xlabel("Time/s")
# plt.ylabel("Charge/nC")


# proper timing run ----------------------------------
# # timing order:
# # 1. time before shaker
# # 2. shaker time
# # 3. time after shaker
# # 4. measurement is taken
# # 5. repeat
bead1_times_full = np.array([
    0, 0, 0,  # after 0 is measure
    28, 60, 17,  # after 17 is measure
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 65, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    28, 60, 17,
    # 28, 0, 4 * 60 + 17,
    # 28, 60, 17
])
bead1_times_sum = np.cumsum(bead1_times_full)
bead1_times = np.zeros(int(np.size(bead1_times_sum) / 3))
i = -2
for element in bead1_times_sum:
    if i % 3 == 0 and i >= 0:
        bead1_times[int(i / 3)] = element  # every 3rd value
    i += 1
bead1_charges = np.array([
    -0.082,
    0.169,
    0.428,
    0.654,
    0.836,
    0.914,
    1.063,
    1.192,
    1.234,
    1.270,
    1.274,
    1.312,
    1.264,
    1.316,
    1.303,
    1.312,
    # 1.188,
    # 1.234
])

fig_c2 = plt.figure()
fig_c2.canvas.manager.set_window_title("Charge against time")
plt.plot(bead1_times, bead1_charges, 'rx')
plt.xlabel("Real elapsed time/s")
plt.ylabel("Charge/nC")


# proper timing run ----------------------------------
# bead1_times_full = np.a([
#     0, 0, 0,
#     60 + 37, 60, 25,
#     31, 60, 24,
#     35, 60, 21,
#     31, 60, 20,
#     31, 60, 22,
#     31, 60, 16,
#     37, 60, 19,
#     34, 120, 18,
#     34, 120, 19
# ])
# bead1_times_sum = np.cumsum(bead1_times_full)
# bead1_times = np.zeros(int(np.size(bead1_times_sum) / 3))
# i = -2
# for element in bead1_times_sum:
#     if i % 3 == 0 and i >= 0:
#         bead1_times[int(i / 3)] = element  # every 3rd value
#     i += 1
# bead1_charges = np.a([
#     0.009,
#     0.226,
#     0.397,
#     0.513,
#     0.449,
#     0.399,
#     0.573,
#     0.691,
#     0.580,
#     0.495
# ])
#
# fig_c3 = plt.figure()
# fig_c3.canvas.manager.set_window_title("Charge against time")
# plt.plot(bead1_times, bead1_charges, 'rx',
#          [bead1_times_sum, bead1_times_sum], [np.ones(np.shape(bead1_times_sum)) * np.amax(bead1_charges),
#                                               np.ones(np.shape(bead1_times_sum)) * np.amin(bead1_charges)],
#          'g--')
# plt.xlabel("Real elapsed time/s")
# plt.ylabel("Charge/nC")

# proper timing run ----------------------------------
# # bead1_times_full = np.a([
# #     0, 0, 0,
# #     27, 0, 18,
# #     30, 0, 18,
# #     34, 0, 24
# # ])
# bead1_times_full = np.a([
#     0, 0, 0,
#     27, 0, 18,
#     30, 0, 18,
#     34, 0, 24,
#     0, 0, 2 * 60,
#     0, 0, 2 * 60,
#     0, 0, 24,
#     0, 0, 39,
#     0, 0, 43
# ])
# bead1_times_sum = np.cumsum(bead1_times_full)
# bead1_times = np.zeros(int(np.size(bead1_times_sum) / 3))
# i = -2
# for element in bead1_times_sum:
#     if i % 3 == 0 and i >= 0:
#         bead1_times[int(i / 3)] = element  # every 3rd value
#     i += 1
# bead1_charges = np.a([
#     0.653,
#     0.632,
#     0.540,
#     0.469,
#     0.412,
#     0.389,
#     0.387,
#     0.369,
#     0.442
# ])
#
# fig_c4 = plt.figure()
# fig_c4.canvas.manager.set_window_title("Charge against time")
# plt.plot(bead1_times, bead1_charges, 'rx',
#          [bead1_times_sum, bead1_times_sum], [np.ones(np.shape(bead1_times_sum)) * np.amax(bead1_charges),
#                                               np.ones(np.shape(bead1_times_sum)) * np.amin(bead1_charges)],
#          'g--')
# plt.xlabel("Real elapsed time/s")
# plt.ylabel("Charge/nC")


# proper timing run ----------------------------------
# bead1_times_full = np.array([
#     0, 0, 0,
#     # 27, 0, 18,
#     30, 0, 18,
#     34, 0, 24,
#     0, 0, 2 * 60,
#     0, 0, 2 * 60,
#     0, 0, 24,
#     # 0, 0, 39,
#     # 0, 0, 43
# ])
# bead1_times_sum = np.cumsum(bead1_times_full)
# bead1_times = np.zeros(int(np.size(bead1_times_sum) / 3))
# i = -2
# for element in bead1_times_sum:
#     if i % 3 == 0 and i >= 0:
#         bead1_times[int(i / 3)] = element  # every 3rd value
#     i += 1
# bead1_charges = np.array([
#     # 0.653,
#     0.632,
#     0.540,
#     0.469,
#     0.412,
#     0.389,
#     0.387,
#     # 0.369,
#     # 0.442
# ])
#
# opt = curve_fit(lambda t, q, k: (bead1_charges[0] - q) * np.exp(k * t) + q,
#                 bead1_times, bead1_charges, p0=(0.38, -0.01))[0]
# offset = opt[0]
# time_constant = opt[1]
# print(f"{offset = :.5g}\n{time_constant = :.5g}")
#
# fig_c5 = plt.figure()
# fig_c5.canvas.manager.set_window_title("Charge against time")
# more_times = np.arange(start=bead1_times[0], stop=bead1_times[-1], step=(bead1_times[-1] - bead1_times[0])/100)
# plt.plot(bead1_times, bead1_charges, 'rx',
#          more_times, (bead1_charges[0] - offset) * np.exp(time_constant * more_times) + offset, 'b-')
# plt.xlabel("Real elapsed time/s")
# plt.ylabel("Charge/nC")

# show plots ----------------------------------
plt.show()
