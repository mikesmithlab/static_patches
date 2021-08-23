import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


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


def my_cross(v1, v2):  # returns cross product of v1 and v2 (for some reason this is a about 20x faster than np.cross!)
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


def find_rotation_matrix(new_x, new_z):  # returns rotation matrix to rotate an object into the new coordinates given
    return np.array([new_x, my_cross(new_x, new_z), new_z])
    # this link laughs in my face
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


def sphere_points_maker(n, offset):  # returns n points on a unit sphere (roughly) evenly distributed
    indices = np.arange(0, n, 1) + offset
    # https://newbedev.com/evenly-distributing-n-points-on-a-sphere
    phi = np.arccos(1 - indices.dot(2 / n))
    theta = indices.dot(np.pi * (1 + 5 ** 0.5))
    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


def offset_finder(n):
    best_std = None
    best_offset = None
    for offset in tqdm(np.arange(0.4, 0.6, 0.00001)):  # offset is in range 0 to 1
        points = sphere_points_maker(n, offset)
        tree = KDTree(points)
        # dists = np.zeros(n)
        # for i in range(n):
        # x = np.ma.array(points[:, 0], mask=False)
        # y = np.ma.array(points[:, 1], mask=False)
        # z = np.ma.array(points[:, 2], mask=False)
        # x.mask[i] = True
        # y.mask[i] = True
        # z.mask[i] = True
        # x = x.compressed()
        # y = y.compressed()
        # z = z.compressed()
        # que = np.array([x, y, z]).T
        dists = np.amax(tree.query(points, k=2)[0], 1)
        standard_deviation = dists.std()
        if best_std is None:
            best_offset = offset
            best_std = standard_deviation
        elif best_std > standard_deviation:
            best_offset = offset
            best_std = standard_deviation
    return best_offset