import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


def q_conjugate(q):  # return the conjugate of a quaternion
    return q[0], -q[1], -q[2], -q[3]


def qq_multiply(q1, q2):  # return the result of quaternion multiplication q1 by q2
    return (q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3],
            q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1])


def qvq_multiply(rotation_quaternion, vector):  # return vector rotated by rotation_quaternion using qvq' multiplication
    # the quaternion (rotation_quaternion) should be unitary (aka normalised)
    return qq_multiply(qq_multiply(rotation_quaternion, [0, *vector]), q_conjugate(rotation_quaternion))[1:]
    # this would output a quaternion but we only want the vector part of it so we have the [1:] at the end


def rotate(rotation, vector):  # returns vector after being rotated by angles around x y and z axes (using quaternions)
    rotation_magnitude = find_magnitude(rotation)
    # todo do a try except here? check speed vs (if magnitude == 0)
    # try:
    #     return np.array(qvq_multiply([
    #         np.cos(rotation_magnitude / 2), *(rotation.dot(np.sin(rotation_magnitude / 2) / rotation_magnitude))],
    #         vector))
    # except ZeroDivisionError:
    #     return vector
    if rotation_magnitude == 0:  # if there isn't a rotation applied, don't rotate!
        return vector
    return np.array(qvq_multiply([
        np.cos(rotation_magnitude / 2), *(rotation.dot(np.sin(rotation_magnitude / 2) / rotation_magnitude))], vector))


def find_magnitude(vector):  # returns magnitude of vector (very fast magnitude finder for small 1D numpy arrays)
    return vector.dot(vector) ** 0.5


def normalise(vector):  # returns vector with magnitude 1 (vector's directional components)
    # todo do a try except here? check speed vs (if magnitude == 0 or magnitude == 1)
    # try:
    #     return vector.dot(1 / find_magnitude(vector))
    # except ZeroDivisionError:
    #     return vector
    magnitude = find_magnitude(vector)
    if magnitude == 0 or magnitude == 1:
        return vector
    return vector.dot(1 / magnitude)


def my_cross(v1, v2):  # returns cross product of v1 and v2 (20x faster than np.cross for small 1D numpy arrays)
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


def find_rotation_matrix(new_x, new_z):  # returns rotation matrix to rotate an object into the new coordinates given
    return np.array([new_x, my_cross(new_x, new_z), new_z])


def find_in_new_coordinates(a, new_x, new_z):  # returns a in new coordinates given by new_x and new_z
    # input a is a np.array of vectors, shaped like (n, 3)
    # credit for this function: https://stackoverflow.com/questions/22081423#22081723
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
        standard_deviation = np.amax(KDTree(points).query(points, k=2)[0], 1).std()  # workers=?
        if best_std is None:
            best_offset = offset
            best_std = standard_deviation
        elif best_std > standard_deviation:
            best_offset = offset
            best_std = standard_deviation
    return best_offset  # standard deviation is not what we want, we want area!!!!


def find_tangent_force(gamma_t, mu, normal_force, normal, surface_velocity):  # returns tangent (friction) force
    tangent_surface_velocity = surface_velocity - normal.dot(normal.dot(surface_velocity))
    try:
        return tangent_surface_velocity.dot(
            -min(gamma_t, mu * find_magnitude(normal_force) / find_magnitude(tangent_surface_velocity)))
    except ZeroDivisionError:  # precisely zero magnitude tangential surface relative velocity causes divide by 0 error
        return np.array([0, 0, 0])


def round_sig_figs(x, p):  # credit for this significant figures function https://stackoverflow.com/revisions/59888924/2
    # x = np.asarray(x)  # can remove this line?
    mags = 10 ** (p - 1 - np.floor(np.log10(np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1)))))
    return np.round(x * mags) / mags
