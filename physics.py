import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from my_tools import find_magnitude, rotate, normalise, find_rotation_matrix, my_cross, sphere_points_maker,\
    find_tangent_force, charge_decay_function, charge_hit_function, round_sig_figs


class PatchTracker:
    """
    Finds which patch a collision happens in and stores the patch numbers with the time
    """

    def __init__(self, n, offset):
        self.tree = KDTree(sphere_points_maker(n, offset))  # input to KDTree for 3D should have dimensions (n, 3)

        self.patches_file = open("patches", "w")
        self.patches_file.writelines(
            "Format: alternating lines, first is time of collision, second is patch indexes of hit"
        )

        self.charges_file = open("charges", "w")
        self.charges_file.writelines(
            "Format: 3 alternating lines, first is time of collision, second is particle patches, "
            "third is container patches"
        )
        self.charges_part = np.zeros(n)
        self.charges_cont = np.zeros(n)

    def collision_update(self, t, pos, particle_x, particle_z):  # input pos: normalised position relative to container
        # find the indexes of the patches that collided on the particle and container
        part = self.tree.query(find_rotation_matrix(particle_x, particle_z).dot(pos), k=1)[1]
        cont = self.tree.query(pos, k=1)[1]
        # output of query is [distance_to_nearest_point, point_number] so we only care about output 1

        # patch tracking for animation and analysis
        self.patches_file.writelines(f"\n{t}\n{part},{cont}")

        # charge tracking for physics
        self.charges_part[part], self.charges_cont[cont] = charge_hit_function(self.charges_part[part],
                                                                               self.charges_cont[cont])

    def close(self):
        self.patches_file.close()
        self.charges_file.close()


class Container:
    """
    Manages container properties and dynamics
    """

    def __init__(self, container_amplitude, container_omega, container_time_end):
        self.container_amplitude = container_amplitude
        self.container_omega = container_omega
        self.container_time_end = container_time_end
        self.container_amplitude_by_omega = self.container_amplitude * self.container_omega

    def container_pos(self, t):  # gives the height of the floor at time t with amplitude a and frequency k
        if t >= self.container_time_end:
            return np.array([0, 0, 0])
        return np.array([0, 0, self.container_amplitude * np.sin(self.container_omega * t)])

    def container_velocity(self, t):  # gives the speed of the floor at time t
        if t >= self.container_time_end:
            return np.array([0, 0, 0])
        return np.array([0, 0, self.container_amplitude_by_omega * np.cos(self.container_omega * t)])


class Particle:
    """
    Manages particle properties, distances, forces, and integration
    """

    def __init__(self, conds, step, g):
        self.radius = conds["radius"]
        self.density = conds["density"]
        self.mass = conds["mass"]
        self.moment_of_inertia = conds["moment_of_inertia"]
        self.spring_constant = conds["spring_constant"]
        self.damping = conds["damping"]
        self.pos = conds["pos"]
        self.velocity = conds["velocity"]
        self.x_axis = normalise(np.array(
            [np.exp(0.5345), np.sqrt(0.456), np.pi / 11]))  # some random numbers so it isn't along any particular axis
        self.z_axis = normalise(my_cross(np.array([0, 0, 1]), self.x_axis))
        self.omega = conds["omega"]

        self.force_multiplier = step / (2 * self.mass)
        self.torque_multiplier = step / (2 * self.moment_of_inertia)
        self.gravity_force = np.array([0, 0, g]).dot(self.mass)

    def find_energy(self, overlap):  # returns the energy of the particle
        return (-self.gravity_force.dot(self.pos) +
                0.5 * self.mass * self.velocity.dot(self.velocity) +
                0.5 * self.spring_constant * overlap ** 2 +
                0.5 * self.moment_of_inertia * self.omega.dot(self.omega))

    def integrate_half(self, time_step, force, torque, first_call):
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)
        if first_call:
            self.pos = self.pos + self.velocity.dot(time_step)
            angles = self.omega.dot(time_step)
            self.x_axis = rotate(angles, self.x_axis)
            self.z_axis = rotate(angles, self.z_axis)


class Engine:
    """
    Runs the physics loop and stores the results in the data_dump file
    """

    def __init__(self, conds):
        self.g = conds["g"]
        self.time_end = conds["time_end"]
        self.time_step = conds["time_step"]
        self.total_steps = conds["total_steps"]
        self.store_interval = conds["store_interval"]
        self.p = Particle(conds, self.time_step, self.g)
        self.c = Container(conds["container_amplitude"], conds["container_omega"], conds["container_time_end"])
        self.p_t = PatchTracker(conds["number_of_patches"], conds["optimal_offset"])
        self.radii_difference = conds["container_radius"] - self.p.radius
        self.mu = conds["mu"]  # coefficient of friction between the surfaces
        self.gamma_t = conds["gamma_t"]  # viscous damping coefficient of the surfaces
        self.contact = False  # todo the only thing contact is used for is graphics. Semi-redundant: patches tracks hits
        self.is_new_collision = True
        decay_constant = -0.01  # approximate decay constant from experimental data. The half life is around 11.5 mins
        self.decay_to_charge = 0.38 * 1e-9  # from experimental data
        self.decay = np.exp(decay_constant * self.time_step / 2)  # (t/2) as its called twice per step

        self.data_file = open("data_dump", "w")  # todo check whether it is stored in memory
        with open("conds.txt", "r") as defaults:
            self.data_file.writelines(defaults.read())
        self.data_file.writelines(
            "\n(end of)iteration,time,pos_x,pos_y,pos_z,"
            "x_axis_x,x_axis_y,x_axis_z,z_axis_x,z_axis_y,z_axis_z,"
            "container_pos,energy,contact"
        )

    def run(self):  # runs the physics loop then cleans up after itself
        for i in tqdm(range(self.total_steps)):  # tqdm gives a progress bar
            time = i * self.time_step
            if i % self.store_interval == 0:  # store if this is a store step
                force, torque, overlap = self.update(time, True)
                self.store(i, time, overlap)
                if i % (self.store_interval * 100) == 0:  # store patch charges every 100 store steps
                    self.p_t.charges_file.writelines(
                        f"\n{time}"
                        f"\n{str(list(round_sig_figs(self.p_t.charges_part, 5))).replace('[', '').replace(']', '')}"
                        f"\n{str(list(round_sig_figs(self.p_t.charges_cont, 5))).replace('[', '').replace(']', '')}"
                    )
                    self.p.x_axis = normalise(self.p.x_axis)
                    self.p.z_axis = normalise(self.p.z_axis)  # todo does this make them innacurate?
            else:  # this else exists for speed - the code runs about 10% faster when overlap isn't assigned in update!
                force, torque, _ = self.update(time, True)
            self.p.integrate_half(self.time_step, force, torque, True)
            force, torque, _ = self.update(time, False)
            self.p.integrate_half(self.time_step, force, torque, False)
        self.close()

    def update(self, t, first_call):  # returns force and torque, also updates distances and patches
        # ----------------
        # charge decay/spreading
        self.p_t.charges_part, self.p_t.charges_cont = charge_decay_function(
            self.p_t.charges_part, self.p_t.charges_cont, self.decay, self.decay_to_charge)
        # ----------------
        # distances
        relative_pos = self.p.pos - self.c.container_pos(t)
        overlap = self.radii_difference - find_magnitude(relative_pos)
        if overlap >= 0:  # overlap is the distance the particle is inside the container wall (is >= 0 if not inside)
            self.contact = False
            if overlap >= self.p.radius * 0.001:
                self.is_new_collision = True
            return self.p.gravity_force, np.array([0, 0, 0]), 0  # return 0 overlap so find_energy doesn't need logic
        self.contact = True
        normal = normalise(relative_pos)
        # ----------------
        # speeds
        overlap_velocity = self.p.velocity - self.c.container_velocity(t)
        surface_relative_velocity = overlap_velocity - my_cross(normal, self.p.omega.dot(self.p.radius))
        # ----------------
        # forces
        normal_force = normal.dot(self.p.spring_constant * overlap - self.p.damping * normal.dot(overlap_velocity))
        tangent_force = find_tangent_force(normal_force, normal, surface_relative_velocity, self.gamma_t, self.mu)
        # ----------------
        # patches
        if first_call and self.is_new_collision:
            self.p_t.collision_update(t, normal, self.p.x_axis, self.p.z_axis)
            self.is_new_collision = False
            # make sure the next calls don't update the patches unless it is a new collision
            # todo this biases the first patch touched (bias direction -avg_angular_velocity)
        return self.p.gravity_force + normal_force + tangent_force, my_cross(
            normal.dot(self.p.radius), tangent_force), overlap

    def store(self, j, t, overlap):  # stores anything that needs storing this step in the data_dump file
        self.data_file.writelines(
            f"\n{j},{t},{self.p.pos[0]:.5g},{self.p.pos[1]:.5g},{self.p.pos[2]:.5g},"
            f"{self.p.x_axis[0]:.5g},{self.p.x_axis[1]:.5g},{self.p.x_axis[2]:.5g},"
            f"{self.p.z_axis[0]:.5g},{self.p.z_axis[1]:.5g},{self.p.z_axis[2]:.5g},"
            f"{self.c.container_pos(t)[2]:.5g},{self.p.find_energy(overlap)},{self.contact}"
        )  # only store container height! container x and y can come later if needed
        # todo currently 5 significant figures, could do 4? or even 3?

    def close(self):  # ensures files are closed when they need to be
        self.data_file.close()
        self.p_t.close()
