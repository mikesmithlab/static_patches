import numpy as np
import matplotlib.pyplot as plt


def show_plots():
    plt.show()


def plot_energy(time_end, total_store):  # produces a plot of energy over time as from the data_dump file
    time_list = np.linspace(0, time_end, num=total_store)
    energy_list = np.zeros(np.shape(time_list))
    try:
        data_file = open("data_dump", "r")
    except FileNotFoundError:
        raise FileNotFoundError("You deleted the data_dump file or didn't make it with Engine")

    i = -3  # set to -3 to get out of the way of the first few lines of non-data
    for line in data_file:
        if i >= 0:
            energy_list[i] = float(line.strip().split(",")[12])  # read energy from this line and store it
        i += 1
    data_file.close()

    fig_e = plt.figure()
    fig_e.canvas.manager.set_window_title("Energy Plot")
    plt.plot(time_list, energy_list)
    plt.xlabel("Time/s")
    plt.ylabel("Energy/J")


def plot_patches(n):  # produces a plot of cumulative hits over time for every patch
    try:
        with open("patches", "r") as patch_file:
            plot_length = int(1 + (len(patch_file.readlines()) - 1) / 2)
            hit_time_list = np.zeros(plot_length)
            p_patch_hit_list = np.zeros([plot_length, n])
            c_patch_hit_list = np.zeros([plot_length, n])

        with open("patches", "r") as patch_file:
            i = -1  # set to -1 to get out of the way of the first line of non-data
            for line in patch_file:
                if i >= 0:
                    if i % 2 == 0:
                        hit_time_list[int(1 + i / 2)] = float(line)  # store the time of this collision
                    else:
                        j = int(1 + (i - 1) / 2)
                        field = line.strip().split(",")
                        p_patch_hit_list[j, :] = p_patch_hit_list[j - 1, :]  # cumulative
                        p_patch_hit_list[j, int(field[0])] += 1  # add one to the number of collisions this patch has
                        c_patch_hit_list[j, :] = c_patch_hit_list[j - 1, :]
                        c_patch_hit_list[j, int(field[1])] += 1
                i += 1
    except FileNotFoundError:
        raise FileNotFoundError("You deleted the patches file or didn't make it with PatchTracker")

    fig_p = plt.figure()
    fig_p.canvas.manager.set_window_title("Particle Patch Hits")
    plt.plot(hit_time_list, p_patch_hit_list, '-',
             hit_time_list, np.sum(p_patch_hit_list, axis=1) / n, '--')  # include average plot
    plt.xlabel("Time/s")
    plt.ylabel("Hits")
    fig_c = plt.figure()
    fig_c.canvas.manager.set_window_title("Container Patch Hits")
    plt.plot(hit_time_list, c_patch_hit_list, '-',
             hit_time_list, np.sum(c_patch_hit_list, axis=1) / n, '--')  # include average plot
    plt.xlabel("Time/s")
    plt.ylabel("Hits")


def plot_charges(n):  # produces a plot of cumulative hits over time for every patch
    try:
        with open("charges", "r") as charge_file:
            plot_length = int(1 + (len(charge_file.readlines()) - 2) / 3)
            time_list = np.zeros(plot_length)
            p_charge_list = np.zeros([plot_length, n])
            c_charge_list = np.zeros([plot_length, n])

        with open("charges", "r") as charge_file:
            i = -1  # set to -1 to get out of the way of the first line of non-data
            for line in charge_file:
                if i >= 0:
                    if i % 3 == 0:
                        time_list[int(i / 3)] = float(line)  # store the time of this collision
                    elif i % 3 == 1:
                        j = int((i - 1) / 3)
                        field = line.strip().split(",")
                        p_charge_list[j, :] = np.array(field)
                    else:
                        j = int((i - 2) / 3)
                        field = line.strip().split(",")
                        c_charge_list[j, :] = np.array(field)
                i += 1
    except FileNotFoundError:
        raise FileNotFoundError("You deleted the charges file or didn't make it with PatchTracker")

    fig_pc = plt.figure()
    fig_pc.canvas.manager.set_window_title("Particle Charge")
    plt.plot(time_list, p_charge_list, time_list, np.sum(p_charge_list, axis=1))  # include total plot
    plt.xlabel("Time/s")
    plt.ylabel("Charge/nC")
    fig_cc = plt.figure()
    fig_cc.canvas.manager.set_window_title("Container Charge")
    plt.plot(time_list, c_charge_list, time_list, np.sum(c_charge_list, axis=1))  # include total plot
    plt.xlabel("Time/s")
    plt.ylabel("Charge/nC")
