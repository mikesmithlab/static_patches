import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from main import find_magnitude, rotate, normalise, find_rotation_matrix, my_cross, sphere_points_maker


class Container:
    """
    Manages container properties and dynamics
    """

    def __init__(self, container_amplitude, container_omega, container_time_end):
        self.container_amplitude = container_amplitude
        self.container_omega = container_omega
        self.container_time_end = container_time_end
        self.container_amplitude_by_omega = self.container_amplitude * self.container_omega

    def container_height(self, t):  # gives the height of the floor at time t with amplitude a and frequency k
        if t >= self.container_time_end:
            return 0
        return self.container_amplitude * np.sin(self.container_omega * t)

    def container_speed(self, t):  # gives the speed of the floor at time t
        if t >= self.container_time_end:
            return 0
        return self.container_amplitude_by_omega * np.cos(self.container_omega * t)


class ParticlePatches:
    """
    Finds which patch a collision happens in and stores the patch number with the time
    """

    def __init__(self, n):
        self.tree = KDTree(sphere_points_maker(n))  # input to KDTree for 3D should have dimensions (n, 3)

        self.patches_file = open("patches", "w")
        first_line = "Format: alternating lines, first is iteration number of collision, second is patch index of hit"
        self.patches_file.writelines(first_line)

    def patch_tracker(self, t, pos, particle_x, particle_z):  # input pos is position relative to container
        write = f"\n{t}\n{self.tree.query(find_rotation_matrix(particle_x, particle_z).dot(normalise(pos)), k=1)[1]}"
        # output of query is [distance_to_nearest_point, point_number] so we only care about output 1
        self.patches_file.writelines(write)

    def close(self):
        self.patches_file.close()


class Particle:
    """
    Manages particle properties, distances, forces, and integration
    """

    def __init__(self, conds, step, g):
        self.p_p = ParticlePatches(conds["number_of_patches"])
        self.c = Container(conds["container_amplitude"], conds["container_omega"], conds["container_time_end"])

        self.radius, self.density = conds["radius"], conds["density"]
        self.mass, self.moment_of_inertia = conds["mass"], conds["moment_of_inertia"]
        self.spring_constant, self.damping = conds["spring_constant"], conds["damping"]
        self.pos = conds["pos"]
        self.velocity = conds["velocity"]
        self.particle_x = normalise(np.array(
            [np.exp(0.5345), np.sqrt(0.456), np.pi / 11]))  # some random numbers so it isn't along any particular axis
        self.particle_z = normalise(my_cross(np.array([0, 0, 1]), self.particle_x))
        self.omega = conds["omega"]
        self.mu, self.gamma_t = conds["mu"], conds["gamma_t"]
        self.radii_difference = conds["container_radius"] - self.radius

        self.force_multiplier = step / (2 * self.mass)
        self.torque_multiplier = step / (2 * self.moment_of_inertia)
        self.gravity_force = np.array([0, 0, g]).dot(self.mass)

        self.overlap, self.overlap_speed = 0, 0
        self.contact = False  # todo the only thing contact is used for is graphics. Semi-redundant: patches tracks hits
        self.is_new_collision = True

    def update(self, t, do_patches):  # returns force and torque, also updates distances and patches
        # distances
        relative_pos = self.pos - np.array([0, 0, self.c.container_height(t)])
        self.find_overlap(relative_pos)
        if self.overlap >= 0:
            self.contact = False
            self.is_new_collision = True
            return self.gravity_force, np.array([0, 0, 0])
        self.contact = True
        # patches
        # todo do container patches as well
        if do_patches and self.is_new_collision:
            self.p_p.patch_tracker(t, relative_pos, self.particle_x, self.particle_z)
            self.is_new_collision = False
            # make sure the next calls don't update the patches unless it is a new collision
            # todo this biases the first patch touched (bias direction -avg_angular_velocity)
        # forces
        normal_force, normal = self.find_normal_force(relative_pos, t)
        tangent_force = self.find_tangent_force(normal_force, normal)
        return self.gravity_force + normal_force + tangent_force, my_cross(normal.dot(self.radius), tangent_force)

    def find_overlap(self, relative_pos):  # finds the distance that the particle is inside the container wall
        self.overlap = self.radii_difference - find_magnitude(relative_pos)
        # todo should this be a function or just a line in p.update?

    def find_normal_force(self, relative_pos, t):  # returns the normal force (and direction) using spring force
        normal = normalise(relative_pos)
        self.overlap_speed = self.velocity - np.array([0, 0, self.c.container_speed(t)])
        return normal.dot(self.spring_constant * self.overlap - self.damping * normal.dot(self.overlap_speed)), normal

    def find_tangent_force(self, normal_contact_force, normal):  # returns the tangent force caused by friction
        surface_relative_velocity = self.overlap_speed - my_cross(normal, self.omega.dot(self.radius))
        tangent_surface_relative_velocity = surface_relative_velocity - normal.dot(surface_relative_velocity) * normal
        xi_dot = find_magnitude(tangent_surface_relative_velocity)
        if xi_dot == 0:  # precisely zero magnitude tangential surface relative velocity causes divide by 0 error
            return np.array([0, 0, 0])
        tangent_direction = tangent_surface_relative_velocity.dot(1 / xi_dot)
        return tangent_direction.dot(-min(self.gamma_t * xi_dot, self.mu * find_magnitude(normal_contact_force)))

    def find_energy(self):  # returns the energy of the particle
        return (-self.gravity_force.dot(self.pos) +
                0.5 * self.mass * self.velocity.dot(self.velocity) +
                0.5 * self.spring_constant * self.overlap ** 2 +
                0.5 * self.moment_of_inertia * self.omega.dot(self.omega))
        # todo check the angular energy here - in fact check all of it

    def integrate(self, t, step):  # performs integration on the particle to find new pos and vel at advanced time
        force, torque = self.update(t, True)
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)
        self.pos = self.pos + self.velocity.dot(step)
        angles = self.omega.dot(step)
        self.particle_x = normalise(rotate(angles, self.particle_x))
        self.particle_z = normalise(rotate(angles, self.particle_z))
        force, torque = self.update(t, False)
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)


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

        self.data_file = open("data_dump", "w")
        defaults = open("conds.txt", "r")  # todo small problem: if conds has the wrong format, get_conds ignores it
        self.data_file.writelines(defaults.read())
        defaults.close()
        info_line = (
                "\niteration,time,pos_x,pos_y,pos_z,particle_x_axis_x,particle_x_axis_y,particle_x_axis_z,"
                "particle_z_axis_x,particle_z_axis_y,particle_z_axis_z,container_pos,energy,contact"
        )
        self.data_file.writelines(info_line)

        self.total_store = 0

    def store(self, j, t):  # stores anything that needs storing this step in the data_dump file
        data = (
            f"\n{j},{t},{self.p.pos[0]:.5g},{self.p.pos[1]:.5g},{self.p.pos[2]:.5g},"
            f"{self.p.particle_x[0]:.5g},{self.p.particle_x[1]:.5g},{self.p.particle_x[2]:.5g},"
            f"{self.p.particle_z[0]:.5g},{self.p.particle_z[1]:.5g},{self.p.particle_z[2]:.5g},"
            f"{self.p.c.container_height(t):.5g},{self.p.find_energy()},{self.p.contact}"
        )
        self.data_file.writelines(data)
        self.total_store += 1

    def run(self):  # runs the physics loop then cleans up after itself
        for i in tqdm(range(self.total_steps)):  # tqdm gives a progress bar
            time = i * self.time_step
            if i % self.store_interval == 0:
                self.store(i, time)  # store if this is a store step
            self.p.integrate(time, self.time_step)  # step forwards by numerical integration
        self.close()

    def close(self):  # ensures files are closed when they need to be
        self.data_file.close()
        self.p.p_p.close()
