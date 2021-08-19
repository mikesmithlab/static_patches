import numpy as np
from scipy.spatial import KDTree


# todo
# air resistance? linear and/or rotational ---- do calculations
# patches where there was contact on the container
# measure all physical parameters:
# container_amplitude
# container_radius
# density
# coefficient_of_restitution
# gamma_t (viscous damping coefficient)
# todo optimise for speed (side goal)
# instead of v1 + v2 do np.add(v1, v2) or try the v1.add(v2)? check speed
# todo 0.5? 1? what is the stepping? how exactly does the verlet work
# todo reduce data_dump filesize by limiting precision! don't limit energy but limit everything else
# todo do we ever need to define self.part = p? can we not just use p?


def q_conjugate(q):  # return the conjugate of a quaternion
    w, x, y, z = q
    return w, -x, -y, -z


def qq_multiply(q1, q2):  # return the result of quaternion multiplied by quaternion in the order q1 by q2
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def qvq_multiply(rotation_quaternion, vector):  # return vector rotated by rotation_quaternion using qvq' multiplication
    # the quaternion (rotation_q) should be unitary (aka normalised)
    return qq_multiply(qq_multiply(rotation_quaternion, [0, *vector]), q_conjugate(rotation_quaternion))[1:]
    # this would output a quaternion but we only want the vector part of it so we have the [1:] at the end


def rotate(rotation, vector):  # returns vector after being rotated by euler angles around x y and z (using quaternions)
    rotation_magnitude = find_magnitude(rotation)
    if rotation_magnitude == 0:  # if there isn't a rotation applied, don't rotate!
        return vector
    return np.array(qvq_multiply([
        np.cos(rotation_magnitude / 2), *(rotation.dot(np.sin(rotation_magnitude / 2) / rotation_magnitude))], vector))


def find_magnitude(vector):  # very fast magnitude finder for numpy vectors
    return vector.dot(vector) ** 0.5


def normalise(vector):  # returns vector with magnitude 1 (so its directional components)
    magnitude = find_magnitude(vector)
    if magnitude == 0 or magnitude == 1:  # todo "or magnitude == 1" should I include?
        return vector
    return vector.dot(1 / magnitude)


def find_rotation_matrix(new_x, new_z):  # returns rotation matrix to rotate an object into the new coordinates given
    return np.array([new_x, my_cross(new_x, new_z), new_z])
    # this link laughs in my face
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


def sphere_points_maker(n):  # returns n points on a unit sphere (roughly) evenly distributed
    indices = np.arange(0, n, 1) + 0.5  # todo this 0.5 is not random and must be optimised for each n value
    # https://newbedev.com/evenly-distributing-n-points-on-a-sphere
    phi = np.arccos(1 - indices.dot(2 / n))
    theta = indices.dot(np.pi * (1 + 5 ** 0.5))
    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


def my_cross(v1, v2):  # for some reason crossing like this is a about 20x faster than np.cross???
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


class Container:
    """
    manages container properties and dynamics
    """

    def __init__(self):
        self.container_radius = float(d.getter("container_radius"))
        self.container_amplitude = float(d.getter("container_amplitude"))
        self.container_omega = float(d.getter("container_omega"))
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
        n = int(d.getter("number_of_patches"))
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

        self.radius, self.density = float(d.getter("radius")), float(d.getter("density"))
        self.mass, self.moment_of_inertia = float(d.getter("mass")), float(d.getter("moment_of_inertia"))
        self.spring_constant, self.damping = float(d.getter("spring_constant")), float(d.getter("damping"))
        self.pos = np.array([float(d.getter("pos_x")), float(d.getter("pos_y")), float(d.getter("pos_z"))])
        self.velocity = np.array(
            [float(d.getter("velocity_x")), float(d.getter("velocity_y")), float(d.getter("velocity_z"))])
        self.particle_x = normalise(np.array(
            [np.exp(0.5345), np.sqrt(0.456), np.pi / 11]))  # some random numbers so it isn't along any particular axis
        self.particle_z = normalise(my_cross(np.array([0, 0, 1]), self.particle_x))
        self.omega = np.array([float(d.getter("omega_x")), float(d.getter("omega_y")), float(d.getter("omega_z"))])
        self.mu, self.gamma_t = float(d.getter("mu")), float(d.getter("gamma_t"))
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
        self.g = float(d.getter("g"))
        self.time_end, self.time_step = float(d.getter("time_end")), float(d.getter("time_step"))
        self.store_interval = int(d.getter("store_interval"))
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
        for i in tqdm(range(int(d.getter("total_steps")))):
            time = i * self.time_step
            # store if this is a store step
            if i % self.store_interval == 0:
                e.store(i, time)
            # step forwards by integration
            self.p.integrate(time, self.time_step)

    def close(self):
        self.data_file.close()
        self.p.p_p.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from tqdm import tqdm
    from defaults import Defaults
    from reader import Animator
    from reader import plot_energy, plot_patches

    # from defualts2 import Defs
    # defs = Defs()

    # todo better way of choosing what to do please? True False commenting out is strange

    d = Defaults()
    get_new_conditions = False
    # get_new_conditions = True
    if get_new_conditions:
        d.setter()
        print("is_new_collision initial conditions with new randomness")
        do_physics = True
    else:
        print("kept previous initial conditions")
        do_physics = False
        # do_physics = True

    if do_physics:
        print("doing physics...")
        e = Engine()  # todo input get_new_conditions
        e.run()
        e.close()
        print("physics is done")
    else:
        print("kept previous physics")

    # do_animate = False
    do_animate = True
    if do_animate:
        print("animating....")
        Animator().animate(int(d.getter("total_store")))

    do_analysis = False
    # do_analysis = True
    if do_analysis:
        print("analysing....")
        # do_energy_analysis = False
        do_energy_analysis = True
        # do_patch_analysis = False
        do_patch_analysis = True
        if do_energy_analysis:
            plot_energy(do_patch_analysis)
        if do_patch_analysis:
            plot_patches()
