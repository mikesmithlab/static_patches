import numpy as np


class Defaults:
    """sets and gets default values for variables and constants"""

    def __init__(self):
        try:
            self.defaults_file = open("default_settings", "r")
        except FileNotFoundError:
            self.defaults_file = open("default_settings", "w")
        self.defaults_file.close()

    def setter(self):
        self.defaults_file = open("default_settings", "w")
        g, container_radius, radius, density = -9.81, 20 / 1000, 5 / 1000, 1000
        container_amplitude, container_omega = 2 / 1000, 20 * 2 * np.pi
        mass = density * (4 / 3) * np.pi * radius ** 3
        moment_of_inertia = (2 / 5) * mass * radius ** 2
        spring_constant = (100 * mass / radius) * (
                np.sqrt(np.sum(g ** 2)) + 2 * container_amplitude * container_omega ** 2)
        coefficient_of_restitution = 0.8  # todo find an appropriate coefficient of restitution
        mu = 0.5  # todo find appropriate coefficient of friction
        gamma_t = 0.8  # todo find appropriate viscous damping coefficient
        damping = (-2) * mass * np.sqrt(spring_constant / mass) * np.log(coefficient_of_restitution) / np.pi
        pos = np.array(
            [0.5 * radius * np.random.randn(), 0.5 * radius * np.random.randn(), 0 * (radius - container_radius)])
        velocity = 1 * np.array([0.1 * g * np.random.randn(), 0.1 * g * np.random.randn(), 0])
        omega = 0 * np.array([50 * 2 * np.pi, 5 * 2 * np.pi, 12 * 2 * np.pi])
        time_end, time_step = 50, (1 / 50) * np.pi * np.sqrt((mass / spring_constant))
        total_steps = int(time_end / time_step)
        # total_steps = (t_end / t_step) + ((int(t_end / t_step) - (t_end / t_step))) != 0)  # todo round up integer?
        # store_interval = int((6 / 50) / (time_step * 60))  # 60Hz
        store_interval = int((6 / 50) / (time_step * 144))  # 144Hz
        total_store = int(total_steps / store_interval) + (total_steps % store_interval > 0)  # round up int because of 0th frame
        number_of_patches = int(200)

        first_line = (
            "g,container_radius,radius,density,container_amplitude,container_omega,mass,moment_of_inertia,"
            "spring_constant, coefficient_of_restitution,mu,gamma_t,damping,pos_x,pos_y,pos_z,velocity_x,"
            "velocity_y,velocity_z,omega_x,omega_y,omega_z,time_end,time_step,total_steps,store_interval,"
            "total_store,number_of_patches\n"
        )
        self.defaults_file.writelines(first_line)
        default_values = (
                f"{g},{container_radius},{radius},{density},{container_amplitude},{container_omega},{mass},"
                f"{moment_of_inertia},{spring_constant},{coefficient_of_restitution},{mu},{gamma_t},"
                f"{damping},{pos[0]},{pos[1]},{pos[2]},{velocity[0]},{velocity[1]},{velocity[2]},"
                f"{omega[0]},{omega[1]},{omega[2]},{time_end},{time_step},{total_steps},{store_interval},"
                f"{total_store},{number_of_patches}\n"
        )
        self.defaults_file.writelines(default_values)
        self.defaults_file.close()

    def getter(self, attribute):
        self.defaults_file = open("default_settings", "r")
        line = self.defaults_file.readline()
        line = line.strip()
        field = line.split(",")
        i = 0
        while i < len(field):
            if field[i] == attribute:
                break
            i += 1
        if i == len(field):
            raise ValueError(f"couldn't find a match for {attribute} in default_settings")

        line = self.defaults_file.readline()
        line = line.strip()
        field = line.split(",")
        self.defaults_file.close()
        return field[i]
