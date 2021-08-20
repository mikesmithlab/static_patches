import numpy as np


# todo
# air resistance? linear and/or rotational ---- do calculations
# patches where there was contact on the container
# measure all physical parameters:
# container_amplitude
# container_radius
# density
# coefficient_of_restitution
# gamma_t (viscous damping coefficient)
# mu (coefficient of friction)
# todo optimise for speed (side goal)
# instead of v1 + v2 do np.add(v1, v2) or try the v1.add(v2)? check speed
# todo 0.5? 1? what is the stepping? how exactly does the verlet work
# todo get_conditions should only be imported to main, then put into other classes as inputs
# todo tiny tiny bit of randomness in the collision forces (more realistic)? Do this after checking energy


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


def find_magnitude(vector):  # returns magnitude of vector (very fast magnitude finder for numpy vectors)
    return vector.dot(vector) ** 0.5


def normalise(vector):  # returns vector with magnitude 1 (vector's directional components)
    magnitude = find_magnitude(vector)
    if magnitude == 0 or magnitude == 1:
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


def my_cross(v1, v2):  # returns cross product of v1 and v2 (for some reason this is a about 20x faster than np.cross!)
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from conditions import get_conditions
    conds = get_conditions(filename="conds.txt")

    # todo better way of choosing what to do please? True False commenting out is strange
    do_physics = False
    # do_physics = True
    if do_physics:
        print("doing physics...")
        from objects import Engine
        Engine(conds).run()
        print("physics is done - data_dump has been written to")
    else:
        print("kept previous physics - data_dump is unchanged")

    # do_animate = False
    do_animate = True
    if do_animate:
        print("animating....")
        from reader import Animator
        Animator(conds).animate()

    do_analysis = False
    # do_analysis = True
    if do_analysis:
        print("analysing....")
        # do_energy_analysis = False
        do_energy_analysis = True
        # do_patch_analysis = False
        do_patch_analysis = True
        if do_energy_analysis:
            from reader import plot_energy
            plot_energy(do_patch_analysis, conds["time_end"], conds["total_store"])
        if do_patch_analysis:
            from reader import plot_patches
            plot_patches(conds["number_of_patches"])
