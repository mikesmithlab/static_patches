import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from my_tools import find_magnitude, rotate, normalise, find_rotation_matrix, my_cross, sphere_points_maker,\
    find_tangent_force, round_sig_figs, find_in_new_coordinates
from charging_functions import charge_decay_function, charge_hit_function


class PatchTracker:
    """
    Finds which patches a collision happens between, then stores their indexes and updates their charges
    """

    def __init__(self, n, offset, r_part, r_cont):
        self.n = n
        self.points = sphere_points_maker(self.n, offset)  # todo separate n for part and cont?
        self.points_part = self.points.dot(r_part)
        self.points_cont = self.points.dot(r_cont)
        self.tree = KDTree(self.points)  # input to KDTree for 3D should have dimensions (n, 3)
        # self.tree_cont = KDTree(self.points_cont)

        with open("patches", "w") as patches_file:
            patches_file.writelines(
                "Format: alternating lines, first is time of collision, second is patch indexes of hit"
            )

        with open("charges", "w") as charges_file:
            charges_file.writelines(
                "Format: 3 alternating lines, first is time of collision, second is particle patches, "
                "third is container patches"
            )
        self.charges_part = np.ones(self.n) * (0 / self.n)  # (-1e-9 / n)
        self.charges_cont = np.ones(self.n) * (0 / self.n)  # starting charge

    def collision_update(self, t, pos, particle_x, particle_z):  # input pos: normalised position relative to container
        # ----------------
        # find the indexes of the patches that collided on the particle and container
        part, cont = self.tree.query([find_rotation_matrix(particle_x, particle_z).dot(pos), pos], k=1)[1]
        # ----------------
        # patch tracking for animation (and analysis)
        with open("patches", "a") as patches_file:
            patches_file.writelines(f"\n{t}\n{part},{cont}")
        # ----------------
        # charge tracking for physics
        self.charges_part[part], self.charges_cont[cont] = charge_hit_function(self.charges_part[part],
                                                                               self.charges_cont[cont])

    def store_charges(self, time):
        with open("charges", "a") as charges_file:
            charges_file.writelines(
                f"\n{time}"
                f"\n{str(list(round_sig_figs(self.charges_part, 5))).replace('[', '').replace(']', '')}"
                f"\n{str(list(round_sig_figs(self.charges_cont, 5))).replace('[', '').replace(']', '')}"
            )

    def find_electrostatics(self, x, z, pos):  # finds overall electrostatic force and torque on the particle
        rotated_points = find_in_new_coordinates(self.points_part, x, z)
        electrostatic_forces = self.find_electrostatic_forces(rotated_points + pos)
        return np.sum(electrostatic_forces, axis=0), np.sum(np.cross(rotated_points, electrostatic_forces), axis=0)

    def find_electrostatic_forces(self, points_part):  # returns the net electrostatic force on each patch
        difference = self.points_cont[:, :, np.newaxis].reshape(1, 3, self.n) - points_part[:, :, np.newaxis]
        # sum((direction) * (magnitude), sum over container patches) to give force on every patch
        # distance^-1 appears multiple times in the maths but only once here (for code speed) so it may not be clear
        return np.sum(difference * (8.99e9 * self.charges_part[:, np.newaxis]
                                    * self.charges_cont[:, np.newaxis].reshape(1, self.n)
                                    * np.sum(np.square(difference), axis=1) ** -1.5)[:, np.newaxis, :],
                      axis=2)  # sum over all container patch interactions for each particle patch


class Container:
    """
    Manages container properties and dynamics
    """

    def __init__(self, container_amplitude, container_omega, container_time_end, container_radius):
        self.amplitude = container_amplitude
        self.omega = container_omega
        self.time_end = container_time_end
        self.amplitude_by_omega = self.amplitude * self.omega
        self.radius = container_radius

    def container_pos(self, t):  # gives the height of the floor at time t with amplitude a and frequency k
        if t >= self.time_end:
            return np.array([0, 0, 0])
        return np.array([0, 0, self.amplitude * np.sin(self.omega * t)])

    def container_velocity(self, t):  # gives the speed of the floor at time t
        if t >= self.time_end:
            return np.array([0, 0, 0])
        return np.array([0, 0, self.amplitude_by_omega * np.cos(self.omega * t)])


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
        self.electrostatic_force = np.array([0, 0, 0])
        self.electrostatic_torque = np.array([0, 0, 0])

    def find_energy(self, overlap):  # returns the energy of the particle
        return (-self.gravity_force.dot(self.pos) +
                0.5 * self.mass * self.velocity.dot(self.velocity) +
                0.5 * self.spring_constant * overlap ** 2 +
                0.5 * self.moment_of_inertia * self.omega.dot(self.omega))

    def integrate_half(self, force, torque, time_step=None):  # update linear and angular velocities in verlet half-step
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)
        if time_step is not None:  # update linear and angular positions on the first half-step
            self.pos = self.pos + self.velocity.dot(time_step)
            angles = self.omega.dot(time_step)
            self.x_axis = normalise(rotate(angles, self.x_axis))
            self.z_axis = normalise(rotate(angles, self.z_axis))
            # the axis directions need normalising every so often, not every step! This takes computation time


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
        self.c = Container(conds["container_amplitude"], conds["container_omega"], conds["container_time_end"],
                           conds["container_radius"])
        self.p_t = PatchTracker(conds["number_of_patches"], conds["optimal_offset"], self.p.radius, self.c.radius)
        self.radii_difference = self.c.radius - self.p.radius
        self.mu = conds["mu"]  # coefficient of friction between the surfaces
        self.gamma_t = conds["gamma_t"]  # viscous damping coefficient of the surfaces
        self.contact = False  # todo the only thing contact is used for is graphics. Semi-redundant: patches tracks hits
        self.is_new_collision = True
        self.impulse_non_e = np.array([0, 0, 0])
        self.impulse_e = np.array([0, 0, 0])

        with open("data_dump", "w") as data_file:
            with open("conds.txt", "r") as conds_file:
                data_file.writelines(conds_file.read())
            data_file.writelines(
                "\n(end of)iteration,time,pos_x,pos_y,pos_z,"
                "x_axis_x,x_axis_y,x_axis_z,z_axis_x,z_axis_y,z_axis_z,"
                "container_pos,energy,contact"
            )

    def run(self):  # runs the physics loop
        for i in tqdm(range(self.total_steps)):  # tqdm gives a progress bar
            time = i * self.time_step
            if i % self.store_interval == 0:  # store if this is a store step
                force, torque, overlap = self.update(time, True)
                self.store(i, time, overlap)
                if i % (self.store_interval * self.p_t.n) == 0:  # store patch charges every n store steps
                    self.p_t.store_charges(time)
            else:  # this else exists for speed - the code runs about 10% faster when overlap isn't assigned in update!
                force, torque, _ = self.update(time, True)
            self.p.integrate_half(force, torque, time_step=self.time_step)
            force, torque, _ = self.update(time, False)
            self.p.integrate_half(force, torque)

    def update(self, t, first_call):  # returns force and torque, also updates distances and patches
        # ----------------
        # charge decay/spreading
        self.p_t.charges_part = charge_decay_function(self.p_t.charges_part, self.time_step / 2)
        self.p_t.charges_cont = charge_decay_function(self.p_t.charges_cont, self.time_step / 2)
        # (t/2) as its called twice per step
        # ----------------
        # distance(s)
        relative_pos = self.p.pos - self.c.container_pos(t)
        overlap = self.radii_difference - find_magnitude(relative_pos)
        # ----------------
        # charge forces
        if False:
        # if first_call:  # todo remove? slows down but increases accuracy on electrostatics
            self.p.electrostatic_force, self.p.electrostatic_torque = self.p_t.find_electrostatics(
                self.p.x_axis, self.p.z_axis, relative_pos)
        # ----------------
        # check for contact
        if overlap >= 0:  # overlap is the distance the particle is inside the container wall (is >= 0 if not inside)
            if self.contact:
                self.contact = False  # update contact bool
            #     print("----------------")
            #     print(f"non-elec = {find_magnitude(self.impulse_non_e)}")
            #     print(f"electric = {find_magnitude(self.impulse_e)}")
            #     self.impulse_non_e = np.array([0, 0, 0])
            #     self.impulse_e = np.array([0, 0, 0])
            # else:
            #     self.impulse_non_e = self.impulse_non_e + self.p.gravity_force
            #     self.impulse_e = self.impulse_e + self.p.electrostatic_force
            if overlap >= self.p.radius * 1e-3:
                # collisions are only counted if the surfaces have previously had a non-negligible distance between them
                self.is_new_collision = True
            # return 0 overlap so find_energy doesn't need logic
            return self.p.gravity_force + self.p.electrostatic_force, self.p.electrostatic_torque, 0
        self.contact = True
        normal = normalise(relative_pos)
        # ----------------
        # speed(s)
        overlap_vel = self.p.velocity - self.c.container_velocity(t)
        # ----------------
        # contact forces
        normal_contact_force = normal.dot(self.p.spring_constant * overlap - self.p.damping * normal.dot(overlap_vel))
        tangent_contact_force = find_tangent_force(self.gamma_t, self.mu, normal_contact_force, normal,
                                                   overlap_vel - my_cross(normal, self.p.omega.dot(self.p.radius)))
        # ----------------
        # patches
        if first_call and self.is_new_collision:
            self.p_t.collision_update(t, normal, self.p.x_axis, self.p.z_axis)
            self.is_new_collision = False
            # make sure the next calls don't update the patches unless it is a new collision
        return (self.p.gravity_force + normal_contact_force + tangent_contact_force + self.p.electrostatic_force,
                my_cross(normal.dot(self.p.radius), tangent_contact_force) + self.p.electrostatic_torque, overlap)

    def store(self, j, t, overlap):  # stores anything that needs storing this step in the data_dump file
        with open("data_dump", "a") as data_file:
            data_file.writelines(
                f"\n{j},{t},{self.p.pos[0]:.5g},{self.p.pos[1]:.5g},{self.p.pos[2]:.5g},"
                f"{self.p.x_axis[0]:.5g},{self.p.x_axis[1]:.5g},{self.p.x_axis[2]:.5g},"
                f"{self.p.z_axis[0]:.5g},{self.p.z_axis[1]:.5g},{self.p.z_axis[2]:.5g},"
                f"{self.c.container_pos(t)[2]:.5g},{self.p.find_energy(overlap)},{self.contact}"
            )  # only store container height! container x and y can come later if needed
            # currently 5 significant figures, could do 4? or even 3?
