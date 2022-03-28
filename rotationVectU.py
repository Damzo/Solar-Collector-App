import numpy as np
from pandas import array

def rotationVectU(u_vect: np.array, angle: float):
    """
    :param u_vect: input vector [ux, uy, uz], numpy array
    :param angle: rotation angle in radian, float
    :return: rotated vector, numpy array
    """
    
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    ux = u_vect[0]
    uy = u_vect[1]
    uz = u_vect[2]

    rot = np.zeros([3,3])
    rot[:,:] = [[(cos_ang + ux ** 2 * (1 - cos_ang)), (ux * uy * (1 - cos_ang) - uz * sin_ang),
            (ux * uz * (1 - cos_ang) + uy * sin_ang)],
           [(uy * ux * (1 - cos_ang) + uz * sin_ang), (cos_ang + uy ** 2 * (1 - cos_ang)),
            (uy * uz * (1 - cos_ang) - ux * sin_ang)],
           [(uz * ux * (1 - cos_ang) - uy * sin_ang), (uz * uy * (1 - cos_ang) + ux * sin_ang),
            (cos_ang + uz ** 2 * (1 - cos_ang))]]

    return rot
