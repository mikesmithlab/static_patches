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


def q_conjugate(q):
    w, x, y, z = q
    return w, -x, -y, -z
    # this takes basically no time so shouldn't need optimising, unless I overhaul the qvq_multiply func so this would
    # benefit from having [real, -*vector], maybe?


def qq_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
    # (r1, v1)(r2, v2) = (r1 * r2 − v1⋅v2, r1*v2 + r2*v1 + v1×v2) is more concise?
    # todo try the above dot and cross method for speed if qq_multiply takes any time at all
    # qq_multiply currently takes about 1/20 of the total time


def qvq_multiply(rotation_quaternion, vector):  # the quaternion (rotation_q) should be unitary (aka normalised)
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


def normalise(vector):
    magnitude = find_magnitude(vector)
    if magnitude == 0 or magnitude == 1:  # todo "or magnitude == 1" should I include?
        return vector
    return vector.dot(1 / magnitude)


def find_rotation_matrix(new_x, new_z):  # returns rotation matrix to rotate an object into the new coordinates given
    return np.array([new_x, my_cross(new_x, new_z), new_z])
    # this link laughs in my face
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


def sphere_maker(n):
    indices = np.arange(0, n, 1) + 0.5  # todo this 0.5 is not random and must be optimised for each n value
    # https://newbedev.com/evenly-distributing-n-points-on-a-sphere
    phi = np.arccos(1 - indices.dot(2 / n))
    theta = indices.dot(np.pi * (1 + 5 ** 0.5))
    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


def my_cross(v1, v2):  # for some reason crossing like this is a about 20x faster than np.cross???
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


class Container:
    """gives container properties and dynamics"""

    def __init__(self):
        self.container_radius = float(d.getter("container_radius"))
        self.container_amplitude = float(d.getter("container_amplitude"))
        self.container_omega = float(d.getter("container_omega"))
        # self.time_step = float(d.getter("time_step"))
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
    """particle patch tracking"""

    def __init__(self):
        n = int(d.getter("number_of_patches"))
        points = sphere_maker(n)  # todo put straight into KDTree? after debugging it lol
        # todo store points in the same place as I store how many each point gets! :o that'd b pretty cool
        self.tree = KDTree(points)  # points should have dimensions (n, 3)
        self.points_and_hits = np.zeros([n, 4])
        self.points_and_hits[:, :-1] = points  # todo instead of :-1 I could do it to specifically 3D, or do it general
        self.is_new_collision = True
        # todo for speed, move is_new_collision to be in particle? the if can be in particle.

        self.patches_file = open("patches", "w")
        first_line = "Format: alternating lines, first is iteration number of collision, second is patch hit number"
        self.patches_file.writelines(first_line)
        second_line = f"\n{0}\n{','.join([str(element) for element in self.points_and_hits[:, 3].T])}"
        self.patches_file.writelines(second_line)

    def patch_tracker(self, t, pos, particle_x, particle_z):  # input pos is position relative to container
        if self.is_new_collision:
            point_number = self.tree.query(find_rotation_matrix(particle_x, particle_z).dot(normalise(pos)), k=1)[1]
            # output of query is distance, i so we only care about i
            self.points_and_hits[point_number, 3] += 1  # adds 1 to the number of hits for this patch
            write = f"\n{t}\n{','.join([str(element) for element in self.points_and_hits[:, 3].T])}"
            self.patches_file.writelines(write)
            # make sure the next calls don't update the patches unless it is a new collision
            # todo this biases the first patch touched (bias direction -avg_angular_velocity)
            self.is_new_collision = False

    def close(self):
        self.patches_file.close()


class Particle:
    """manages particle properties and forces"""

    def __init__(self):
        self.cont = c
        self.part_patch = p_p

        g = float(d.getter("g"))
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

        self.radii_difference = self.cont.container_radius - self.radius
        self.gravity_force = np.array([0, 0, g]).dot(self.mass)

        self.overlap, self.overlap_speed = 0, 0
        self.contact = False

    def update(self, time, patch):
        # distances
        container_height = self.cont.container_height(time)  # todo do this with container_radius as well? check speed.
        self.find_overlap(container_height)
        if self.overlap >= 0:
            self.contact = False
            self.part_patch.is_new_collision = True
            return self.gravity_force, np.array([0, 0, 0])
        self.contact = True
        # patches
        if patch:
            self.part_patch.patch_tracker(
                time, self.pos - np.array([0, 0, container_height]), self.particle_x, self.particle_z)
            # todo do container patches as well
        # forces
        normal_force, normal = self.find_normal_force(container_height, time)
        tangent_force = self.find_tangent_force(normal_force, normal)
        return self.gravity_force + normal_force + tangent_force, -my_cross(normal.dot(self.radius), tangent_force)

    def find_overlap(self, container_height):
        self.overlap = self.radii_difference - find_magnitude(self.pos - np.array([0, 0, container_height]))

    def find_normal_force(self, container_height, time):
        normal = normalise(self.pos - np.array([0, 0, container_height]))
        self.overlap_speed = self.velocity - np.array([0, 0, self.cont.container_speed(time)])
        return normal.dot(self.spring_constant * self.overlap - self.damping * normal.dot(self.overlap_speed)), normal

    def find_tangent_force(self, normal_contact_force, normal):
        surface_relative_velocity = self.overlap_speed - my_cross(self.omega.dot(self.radius), normal)
        tangent_surface_relative_velocity = surface_relative_velocity - normal.dot(surface_relative_velocity) * normal
        xi_dot = find_magnitude(tangent_surface_relative_velocity)
        if xi_dot == 0:  # precisely zero magnitude tangential surface relative velocity causes divide by 0 error
            return np.array([0, 0, 0])
        tangent_direction = tangent_surface_relative_velocity.dot(1 / xi_dot)
        return tangent_direction.dot(-min(self.gamma_t * xi_dot, self.mu * find_magnitude(normal_contact_force)))


class Engine:
    """integrates and stores"""

    def __init__(self):
        self.cont = c
        self.part = p

        self.time_end, self.time_step = float(d.getter("time_end")), float(d.getter("time_step"))
        self.time_warp = int(d.getter("time_warp"))
        self.force_multiplier = self.time_step / (2 * self.part.mass)
        self.torque_multiplier = self.time_step / (2 * self.part.moment_of_inertia)

        self.data_file = open("data_dump", "w")
        defaults = open("default_settings", "r")
        self.data_file.writelines(defaults.read())
        defaults.close()
        info_line = "iteration," + "time," + "pos_x," + "pos_y," + "pos_z," + \
                    "lump_one_x," + "lump_one_y," + "lump_one_z," + \
                    "lump_two_x," + "lump_two_y," + "lump_two_z," + \
                    "container_pos," + "energy," + "contact"
        self.data_file.writelines(info_line)

        self.total_store = 0

    def single_step(self, j):
        time = j * self.time_step
        if j % self.time_warp == 0:
            self.store(j, time)
        # step forwards by integration
        self.integrate(time)

    def integrate(self, time):
        force, torque = self.part.update(time, True)
        self.part.velocity = self.part.velocity + force.dot(self.force_multiplier)
        self.part.omega = self.part.omega + torque.dot(self.torque_multiplier)
        self.part.pos = self.part.pos + self.part.velocity.dot(self.time_step)
        angles = self.part.omega.dot(-self.time_step)  # todo negative here? why?
        self.part.particle_x = normalise(rotate(angles, self.part.particle_x))
        self.part.particle_z = normalise(rotate(angles, self.part.particle_z))
        force, torque = self.part.update(time, False)
        self.part.velocity = self.part.velocity + force.dot(self.force_multiplier)
        self.part.omega = self.part.omega + torque.dot(self.torque_multiplier)

    def store(self, j, time):
        lump_one = self.part.particle_x
        lump_two = self.part.particle_z
        part_pos = self.part.pos

        energy = (-self.part.gravity_force.dot(part_pos) +
                  0.5 * self.part.mass * self.part.velocity.dot(self.part.velocity) +
                  0.5 * self.part.spring_constant * self.part.overlap ** 2 +
                  0.5 * self.part.moment_of_inertia * self.part.omega.dot(self.part.omega))
        # todo check the angular energy here

        data = f"\n{j},{time},{part_pos[0]},{part_pos[1]},{part_pos[2]},{lump_one[0]},{lump_one[1]},{lump_one[2]}," +\
               f"{lump_two[0]},{lump_two[1]},{lump_two[2]},{self.cont.container_height(time)},{energy}," +\
               f"{self.part.contact}"
        self.data_file.writelines(data)
        self.total_store += 1

    def close(self):
        self.data_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from tqdm import tqdm
    from defaults import Defaults
    from reader import Animator
    from reader import Analyser

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
        c = Container()
        p = Particle()
        e = Engine()
        p_p = ParticlePatches()
        # todo could be a bigger problem with my code, as each time I define p = Particle() I run __init__? slows down
        for i in tqdm(range(int(d.getter("total_steps")))):
            e.single_step(i)
        p_p.close()
        e.close()
        print("physics is done")
    else:
        print("kept previous physics")

    do_animate = False
    # do_animate = True
    if do_animate:
        print("animating....")
        Animator().animate(int(d.getter("total_store")))

    # do_analysis = False
    do_analysis = True
    if do_analysis:
        print("analysing....")
        a = Analyser()
        do_energy = False
        # do_energy = True
        if do_energy:
            a.plot_energy()
        # do_patches = False
        do_patches = True
        if do_patches:
            a.plot_patches()
