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
    # credit for this spreading algorithm: https://newbedev.com/evenly-distributing-n-points-on-a-sphere
    indices = np.arange(0, n, 1) + offset
    phi = np.arccos(1 - indices.dot(2 / n))
    theta = indices.dot(np.pi * (1 + 5 ** 0.5))
    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


def offset_finder(n):  # finds the best offset (within some precision) for the sphere_point_maker for standard deviation
    best_std = None
    best_offset = None
    for offset in tqdm(np.arange(0.4, 0.6, 0.00001)):  # offset is in range 0 to 1
        points = sphere_points_maker(n, offset)
        dists = np.amax(KDTree(points).query(points, k=2)[0], 1)
        standard_deviation = dists.std()
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


def charge_decay_function(charge_part, charge_cont, decay):  # returns the new charges due to decay (to air?)
    # todo:
    # find some correct decay equation
    # does the charge spread to nearby patches? <-- would be horrible to compute
    # decay = np.exp(-0.005 * time_step)  # this decay constant is for half-life of 2 minutes
    return charge_part.dot(decay), charge_cont.dot(decay)


def charge_hit_function(patch_charge_part, patch_charge_cont):  # returns the new charges of colliding patches
    # todo:
    # do previous charges of the patches matter? or just add some constant every collision? ('proper "saturation"')
    # does the force matter?
    # does the charge of nearby patches matter?
    # constant that is added needs to change with patch area (work area out once then input it to this function)
    return patch_charge_part + 1e-13, patch_charge_cont + 1e-13


def round_sig_figs(x, p):  # credit for this significant figures function https://stackoverflow.com/revisions/59888924/2
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
