import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


def q_conjugate(q):  # return the conjugate of a quaternion
    return q[0], -q[1], -q[2], -q[3]
    # w, x, y, z = q
    # return w, -x, -y, -z


def qq_multiply(q1, q2):  # return the result of quaternion multiplied by quaternion in the order q1 by q2
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2)
    # w1, x1, y1, z1 = q1
    # w2, x2, y2, z2 = q2
    # w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    # x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    # y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    # z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    # return w, x, y, z


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


def find_magnitude(vector):  # returns magnitude of vector (very fast magnitude finder for 1D numpy vectors)
    return vector.dot(vector) ** 0.5


def normalise(vector):  # returns vector with magnitude 1 (vector's directional components)
    magnitude = find_magnitude(vector)
    if magnitude == 0 or magnitude == 1:
        return vector
    return vector.dot(1 / magnitude)


def my_cross(v1, v2):  # returns cross product of v1 and v2 (for some reason this is a about 20x faster than np.cross!)
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


def find_rotation_matrix(new_x, new_z):  # returns rotation matrix to rotate an object into the new coordinates given
    return np.array([new_x, my_cross(new_x, new_z), new_z])


def find_in_new_coordinates(a, new_x, new_z):  # returns a in new coordinates given by new_x and new_z
    # input a is a np.array of vectors, shaped like [n, 3]
    # credit for this function goes to
    # https://stackoverflow.com/questions/22081423#22081723
    return np.dot(find_rotation_matrix(new_x, new_z), a.reshape((a.shape[0], a.shape[1])).T).T.reshape(a.shape)


def sphere_points_maker(n, offset):  # returns n points on a unit sphere (roughly) evenly distributed
    # credit for this spreading algorithm: https://newbedev.com/evenly-distributing-n-points-on-a-sphere
    indices = np.arange(0, n, 1) + offset
    phi = np.arccos(1 - indices.dot(2 / n))
    theta = indices.dot(np.pi * (1 + 5 ** 0.5))
    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


def offset_finder(n):  # finds the best offset (within some precision) for the sphere_point_maker for standard deviation
    best_std = None
    best_offset = None
    for offset in tqdm(np.arange(0.4, 0.6, 0.00001)):  # offset is in range 0 to 1, but usually 0.4 to 0.6?
        points = sphere_points_maker(n, offset)
        standard_deviation = np.amax(KDTree(points).query(points, k=2)[0], 1).std()  # todo workers?
        if best_std is None:
            best_offset = offset
            best_std = standard_deviation
        elif best_std > standard_deviation:
            best_offset = offset
            best_std = standard_deviation
    return best_offset  # todo standard deviation is not what we want, we want area!!!!


def find_tangent_force(normal_force, normal, surface_velocity, gamma_t, mu):  # returns tangent (friction) force
    tangent_surface_velocity = surface_velocity - normal.dot(normal.dot(surface_velocity))
    xi_dot = find_magnitude(tangent_surface_velocity)
    if xi_dot == 0:  # precisely zero magnitude tangential surface relative velocity causes divide by 0 error
        return np.array([0, 0, 0])
    tangent_direction = tangent_surface_velocity.dot(1 / xi_dot)
    return tangent_direction.dot(-min(gamma_t * xi_dot, mu * find_magnitude(normal_force)))


def find_electrostatic_forces(charges_part, charges_cont, points_part, points_cont):
    # returns the net electrostatic force on each patch
    # todo minimum distance: at the moment cont radius slightly larger, is this optimal?
    # todo 8.988e9 from wikipedia Coulomb's constant
    difference = (
            points_cont[:, :, np.newaxis]
            - points_part[:, :, np.newaxis].reshape((1, np.shape(points_part)[1], np.shape(points_part)[0]))
    )
    reciprocal_distances = np.reciprocal(np.sum(np.square(difference), axis=1) ** 0.5)[:, np.newaxis, :]
    # reciprocal_distances is the reciprocal of every element of the distances
    # sum((direction) * (magnitude), sum over container patches), then reshape to give force on every patch
    return np.sum(
        (difference * reciprocal_distances)  # (point difference * normalising factor) = direction
        *  # (direction) * (magnitude)
        (
                (charges_part[:, np.newaxis, np.newaxis].reshape(1, 1, np.shape(points_part)[0])
                 *
                 charges_cont[:, np.newaxis, np.newaxis] * 8.988e9)  # (Coulomb's constant * q1 * q2)
                *
                np.square(reciprocal_distances)  # distance^-2
        ), axis=0  # sum over all container patch interactions for each particle patch
    ).reshape(np.shape(points_part)[0], np.shape(points_part)[1])  # final reshape to be (n, 3)
    # --------------------------------
    # the below code is similar vectorised code but assigns variables in multiple steps to be more understandable...
    # however, the code is significantly slower, and uses KDTree for all the distances
    # --------------------------------
    # # shapes:
    # # coord lists: (n, 3)
    # # charge lists: (n, )
    # # charge_forces: (n, 3)
    # the_shape = np.shape(points_part)
    # reciprocal_distances = np.reciprocal(np.array(tree_cont.query(
    #     points_part, k=the_shape[0])[0])[:, :, np.newaxis].reshape(the_shape[0], 1, the_shape[0]))
    # # (reciprocal_)distances (n, n) --> (n, 1, n)
    # kqq = (charges_part[:, np.newaxis, np.newaxis].reshape(1, 1, the_shape[0]) *
    #        charges_cont[:, np.newaxis, np.newaxis] * 8.988e9)
    # # shape kqq (n, 1, n)
    # magnitudes = (kqq * np.square(reciprocal_distances))
    # # shape (n, 1, n)
    # differences = points_cont[:, :, np.newaxis] - points_part[:, :, np.newaxis].reshape(
    #     (1, the_shape[1], the_shape[0]))
    # # difference in position (n, 3, n) with each original patch occupying a slice (dim 2)
    # # directions = differences * np.reciprocal(np.sum(np.square(differences), axis=1) ** 0.5)[:, np.newaxis, :]
    # directions = differences * reciprocal_distances
    # # print(np.shape(directions))
    # charge_forces = np.sum(directions * magnitudes, axis=0).reshape(the_shape[0], the_shape[1])
    # # print(np.shape(charge_forces))
    # return charge_forces  # charge_forces is the net force on each individual patch on part


def charge_decay_function(charge, decay_exponential, shift=0):  # returns the new charges due to decay (to air?)
    # todo does the charge spread to nearby patches? <-- would be horrible to compute
    return (charge - shift).dot(decay_exponential) + shift


def charge_hit_function(patch_charge_part, patch_charge_cont, charge_per_hit):  # returns the new charges of colliding patches
    # todo:
    # do previous charges of the patches matter? or just add some constant every collision? ('proper "saturation"'?)
    # does the force matter?
    # does the charge of nearby patches matter?
    # constant that is added needs to change with patch area (work area out once then input it to this function)
    return patch_charge_part + charge_per_hit, patch_charge_cont - charge_per_hit


def round_sig_figs(x, p):  # credit for this significant figures function https://stackoverflow.com/revisions/59888924/2
    x = np.asarray(x)  # todo can remove this line?
    mags = 10 ** (p - 1 - np.floor(np.log10(np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1)))))
    return np.round(x * mags) / mags
