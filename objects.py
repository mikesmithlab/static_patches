import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from main import opts, find_magnitude, rotate, normalise, find_rotation_matrix, my_cross, sphere_points_maker


class Container:
    """
    manages container properties and dynamics
    """

    def __init__(self):
        self.container_radius = float(opts["container_radius"])
        self.container_amplitude = float(opts["container_amplitude"])
        self.container_omega = 2 * np.pi * float(opts["container_frequency"])
        self.container_amplitude_by_omega = self.container_amplitude * self.container_omega

    def container_height(self, t):  # gives the height of the floor at time t with amplitude a and frequency k
        if t >= 48:
            return 0
        return self.container_amplitude * np.sin(self.container_omega * t)

    def container_speed(self, t):  # gives the speed of the floor at time t
        if t >= 48:
            return 0
        return self.container_amplitude_by_omega * np.cos(self.container_omega * t)


class ParticlePatches:
    """
    manages particle patches
    """

    def __init__(self):
        n = int(opts["number_of_patches"])
        points = sphere_points_maker(n)  # todo put straight into KDTree? after debugging it lol
        self.tree = KDTree(points)  # points should have dimensions (n, 3)

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
    manages particle properties and forces
    """

    def __init__(self, step, g):
        self.p_p = ParticlePatches()
        self.c = Container()

        self.radius, self.density = float(opts["radius"]), float(opts["density"])
        self.mass, self.moment_of_inertia = float(opts["mass"]), float(opts["moment_of_inertia"])
        self.spring_constant, self.damping = float(opts["spring_constant"]), float(opts["damping"])
        self.pos = np.array([float(opts["pos_x"]), float(opts["pos_y"]), float(opts["pos_z"])])
        self.velocity = np.array(
            [float(opts["velocity_x"]), float(opts["velocity_y"]), float(opts["velocity_z"])])
        self.particle_x = normalise(np.array(
            [np.exp(0.5345), np.sqrt(0.456), np.pi / 11]))  # some random numbers so it isn't along any particular axis
        self.particle_z = normalise(my_cross(np.array([0, 0, 1]), self.particle_x))
        self.omega = np.array([float(opts["omega_x"]), float(opts["omega_y"]), float(opts["omega_z"])])
        self.mu, self.gamma_t = float(opts["mu"]), float(opts["gamma_t"])
        self.force_multiplier = step / (2 * self.mass)
        self.torque_multiplier = step / (2 * self.moment_of_inertia)

        self.radii_difference = self.c.container_radius - self.radius
        self.gravity_force = np.array([0, 0, g]).dot(self.mass)

        self.overlap, self.overlap_speed = 0, 0
        self.contact = False
        self.is_new_collision = True

    def update(self, t, do_patches):
        # distances
        container_height = self.c.container_height(t)  # todo do this with container_radius as well? check speed.
        self.find_overlap(container_height)
        if self.overlap >= 0:
            self.contact = False
            self.is_new_collision = True
            return self.gravity_force, np.array([0, 0, 0])
        self.contact = True
        # patches
        # todo do container patches as well
        if do_patches and self.is_new_collision:
            self.p_p.patch_tracker(
                t, self.pos - np.array([0, 0, container_height]), self.particle_x, self.particle_z)
            self.is_new_collision = False
            # make sure the next calls don't update the patches unless it is a new collision
            # todo this biases the first patch touched (bias direction -avg_angular_velocity)
        # forces
        normal_force, normal = self.find_normal_force(container_height, t)
        tangent_force = self.find_tangent_force(normal_force, normal)
        return self.gravity_force + normal_force + tangent_force, -my_cross(normal.dot(self.radius), tangent_force)

    def find_overlap(self, container_height):
        self.overlap = self.radii_difference - find_magnitude(self.pos - np.array([0, 0, container_height]))

    def find_normal_force(self, container_height, t):
        normal = normalise(self.pos - np.array([0, 0, container_height]))
        self.overlap_speed = self.velocity - np.array([0, 0, self.c.container_speed(t)])
        return normal.dot(self.spring_constant * self.overlap - self.damping * normal.dot(self.overlap_speed)), normal

    def find_tangent_force(self, normal_contact_force, normal):
        surface_relative_velocity = self.overlap_speed - my_cross(self.omega.dot(self.radius), normal)
        tangent_surface_relative_velocity = surface_relative_velocity - normal.dot(surface_relative_velocity) * normal
        xi_dot = find_magnitude(tangent_surface_relative_velocity)
        if xi_dot == 0:  # precisely zero magnitude tangential surface relative velocity causes divide by 0 error
            return np.array([0, 0, 0])
        tangent_direction = tangent_surface_relative_velocity.dot(1 / xi_dot)
        return tangent_direction.dot(-min(self.gamma_t * xi_dot, self.mu * find_magnitude(normal_contact_force)))

    def find_energy(self):
        return (-self.gravity_force.dot(self.pos) +
                0.5 * self.mass * self.velocity.dot(self.velocity) +
                0.5 * self.spring_constant * self.overlap ** 2 +
                0.5 * self.moment_of_inertia * self.omega.dot(self.omega))
        # todo check the angular energy here  - in fact check all of it

    def integrate(self, t, step):
        force, torque = self.update(t, True)
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)
        self.pos = self.pos + self.velocity.dot(step)
        angles = self.omega.dot(-step)  # todo negative here? why?
        self.particle_x = normalise(rotate(angles, self.particle_x))
        self.particle_z = normalise(rotate(angles, self.particle_z))
        force, torque = self.update(t, False)
        self.velocity = self.velocity + force.dot(self.force_multiplier)
        self.omega = self.omega + torque.dot(self.torque_multiplier)


class Engine:
    """integrates and stores"""

    def __init__(self):
        self.g = float(opts["g"])
        self.time_end, self.time_step = float(opts["time_end"]), float(opts["time_step"])
        self.store_interval = int(opts["store_interval"])
        self.p = Particle(self.time_step, self.g)

        self.data_file = open("data_dump", "w")
        defaults = open("default_settings", "r")
        self.data_file.writelines(defaults.read())
        defaults.close()
        info_line = (
                "iteration," + "time," + "pos_x," + "pos_y," + "pos_z," +
                "particle_x_axis_x," + "particle_x_axis_y," + "particle_x_axis_z," +
                "particle_z_axis_x," + "particle_z_axis_y," + "particle_z_axis_z," +
                "container_pos," + "energy," + "contact"
        )
        self.data_file.writelines(info_line)

        self.total_store = 0

    def store(self, j, t):
        lump_one = self.p.particle_x
        lump_two = self.p.particle_z
        part_pos = self.p.pos
        # todo these^ might not be any faster
        data = (
            f"\n{j},{t},{part_pos[0]:.5g},{part_pos[1]:.5g},{part_pos[2]:.5g},"
            f"{lump_one[0]:.5g},{lump_one[1]:.5g},{lump_one[2]:.5g},"
            f"{lump_two[0]:.5g},{lump_two[1]:.5g},{lump_two[2]:.5g},"
            f"{self.p.c.container_height(t):.5g},{self.p.find_energy()},{self.p.contact}"
        )
        self.data_file.writelines(data)
        self.total_store += 1

    def run(self):
        for i in tqdm(range(int(opts["total_steps"]))):
            time = i * self.time_step
            # store if this is a store step
            if i % self.store_interval == 0:
                self.store(i, time)
            # step forwards by integration
            self.p.integrate(time, self.time_step)

    def close(self):
        self.data_file.close()
        self.p.p_p.close()
