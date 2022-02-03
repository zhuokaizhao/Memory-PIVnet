# GPU version of vorticity that replaces numpy with cupy
import cupy as cp
import matplotlib.pyplot as plt


def curl(udata, dx=1., dy=1., dz=1., xyz_orientations=cp.asarray([1, -1, 1]),
         xx=None, yy=None, zz=None, verbose=False):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = cp.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy, uz) or (ux, uy)
    dx, dy, dz: float, spatial spating of a 2D/3D grid
    xyz_orientations: 1d array
        ... differentiation in the index space and the physical space must be conducted properly.
        tflow convention is to treat the row, column, and the depth (i,j,k) are parallel to x, y, z in the physical space;
        however, this does not specify the direction of the axes. (+i direction is only equal to EITHER +x or -x).
        This ambiguity causes a problem during differentiation, and the choice is somewhat arbitrary to the users.
        This function offers a solution by two ways. One way is to give a 2d/3d array of the positional grids.
        If xx, yy, zz are given, it would automatically figures out how +i,+j,+k are aligned with +x,+y,+z.
        The second way is give delta x/ delta_i, dy/delta_j, dz/delta_k. This argument is related to this method.
        ... e.g.
        xyz_orientations = [1, 1, 1]... means +x // +i, +y//+j, +z//+k
        xyz_orientations = [-1, -1, -1]... means +x // -i, +y//-j, +z//-k
        xyz_orientations = [1, -1, 1]... means +x // +i, +y//-j, +z//+k
    xx: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +x and +i point at the same direction,
         and the curl is computed based on that
    yy: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +y and +j point at the same direction,
         and the curl is computed based on that
    zz: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +z and +k point at the same direction,
         and the curl is computed based on that

    Returns
    -------
    omega: numpy array
        shape: (height, width, duration) (2D) or (height, width, duration) (2D)

    """
    if verbose:
        print('... curl(): If the result is not sensible, consult altering xyz_orientations.\n'
              'A common mistake is that udata is not origanized properly such that '
              '+x direction is not equal to +direction along the row of an array'
              'or +y direction is not equal to +direction along the column of an array')
    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz, xyz_orientations=xyz_orientations, xx=xx, yy=yy, zz=zz)
    dim = len(sij.shape) - 3  # spatial dim
    eij, gij = decompose_duidxj(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]  # checked. this is correct.
    elif dim == 3:
        # sign issue was checked. this is correct.
        omega1, omega2, omega3 = 2. * gij[..., 2, 1], 2. * gij[..., 0, 2], 2. * gij[..., 1, 0]
        # omega1, omega2, omega3 = -2. * gij[..., 2, 1], 2. * gij[..., 0, 2], -2. * gij[..., 1, 0]
        omega = cp.stack((omega1, omega2, omega3))
    else:
        print('Not implemented yet!')
        return None
    return omega


def get_duidxj_tensor(udata, dx=1., dy=1., dz=1., xyz_orientations=cp.asarray([1, -1, 1]),
                      xx=None, yy=None, zz=None):
    """
    Assumes udata has a shape (d, nrows, ncols, duration) or  (d, nrows, ncols)
    ... one can easily make udata by cp.stack((ux, uy))

    Important Warning:
    ... udata is cp.stack((ux, uy, uz))
    ... udata.shape = dim, nrows, ncols, duration
    Parameters
    ----------
    udata: numpy array with shape (ux, uy) or (ux, uy, uz)
        ... assumes ux/uy/uz has a shape (nrows, ncols, duration) or (nrows, ncols, nstacks, duration)
        ... can handle udata without a temporal axis
    dx: float, x spacing
    dy: float, y spacing
    dz: float, z spacing
    xyz_orientations: 1d array-like with shape (3,)
        ... xyz_orientations = [djdx, didy, dkdz]
        ... HOW TO DETERMINE xyz_orientations:
                1. Does the xx[0, :] (or xx[0, :, 0] for 3D) monotonically increase as the index increases?
                    If True, djdx = 1
                    If False, djdx = -1
                2. Does the yy[:, 0] (or yy[:, 0, 0] for 3D) monotonically increase as the index increases?
                    If True, didy = 1
                    If False, didy = -1
                3. Does the zz[0, 0, :] monotonically increase as the index increases?
                    If True, dkdz = 1
                    If False, dkdz = -1
            ... If you are not sure what this is, use

        ... Factors between index space (ijk) and physical space (xyz)

        ... This factor is necessary since the conventions used for the index space and the physical space are different.
            Consider a 3D array a. a[i, j, k]. The convention used for this module is to interpret this array as a[y, x, z].
            (In case of udata, udata[dim, y, x, z, t])
            All useful modules such as numpy are written in the index space, but this convention is not ideal for physicists
            for several reasons.
            1. many experimental data are not organized in the index space.
            2. One always requires conversion between the index space and physical space especially at the end of the analysis.
            (physicists like to present in the units of physical units not in terms of ijk)

        ... This array is essentially a Jacobian between the index space basis (ijk) and the physical space basis (xyz)
            All information needed is just dx/dj, dy/di, dz/dk because ijk and xyz are both orthogonal bases.
            There is no off-diagonal elements in the Jacobian matrix, and one needs to supply only 3 elements for 3D udata.
            If I strictly use the Cartesian basis for xyz (as it should), then I could say they are both orthonormal.
            This makes each element of the Jacobian array to be either 1 or -1, reflecting the directions of +x/+y/+z
            with respect to +j/+i/+k


    Returns
    -------
    sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
        ... idea is... sij[spacial coordinates, time, tensor indices]
            e.g.-  sij(x, y, t) = sij[y, x, t, i, j]
        ... sij = d ui / dxj
    """

    if xx is not None and yy is not None:
        xyz_orientations = get_jacobian_xyz_ijk(xx, yy, zz)
        if zz is None:
            dx, dy = get_grid_spacing(xx, yy)
        else:
            dx, dy, dz = get_grid_spacing(xx, yy, zz)
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = shape[0]

    if dim == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))

        duxdx = cp.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = cp.gradient(ux, dy, axis=0) * xyz_orientations[
            1]  # +dy is the column up. cp gradient computes difference by going DOWN in the column, which is the opposite
        duydx = cp.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = cp.gradient(uy, dy, axis=0) * xyz_orientations[1]
        sij = cp.zeros((nrows, ncols, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
    elif dim == 3:
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            # print ux.shape
            nrows, ncols, nstacks, duration = ux.shape
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
        duxdx = cp.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = cp.gradient(ux, dy, axis=0) * xyz_orientations[1]
        duxdz = cp.gradient(ux, dz, axis=2) * xyz_orientations[2]
        duydx = cp.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = cp.gradient(uy, dy, axis=0) * xyz_orientations[1]
        duydz = cp.gradient(uy, dz, axis=2) * xyz_orientations[2]
        duzdx = cp.gradient(uz, dx, axis=1) * xyz_orientations[0]
        duzdy = cp.gradient(uz, dy, axis=0) * xyz_orientations[1]
        duzdz = cp.gradient(uz, dz, axis=2) * xyz_orientations[2]

        sij = cp.zeros((nrows, ncols, nstacks, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 0, 2] = duxdz
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
        sij[..., 1, 2] = duydz
        sij[..., 2, 0] = duzdx
        sij[..., 2, 1] = duzdy
        sij[..., 2, 2] = duzdz
    elif dim > 3:
        print('...Not implemented yet.')
        return None
    return sij


def decompose_duidxj(sij):
    """
    Decompose a duidxj tensor into a symmetric and an antisymmetric parts
    Returns symmetric part (eij) and anti-symmetric part (gij)

    Parameters
    ----------
    sij, 5d or 6d numpy array (x, y, t, i, j) or (x, y, z, t, i, j)

    Returns
    -------
    eij: 5d or 6d numpy array, symmetric part of rate-of-strain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.
    gij: 5d or 6d numpy array, anti-symmetric part of rate-of-stxxain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.

    """
    dim = len(sij.shape) - 3  # spatial dim
    if dim == 2:
        duration = sij.shape[2]
    elif dim == 3:
        duration = sij.shape[3]

    eij = cp.zeros(sij.shape)
    # gij = cp.zeros(sij.shape) #anti-symmetric part
    for t in range(duration):
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    eij[..., t, i, j] = 1. / 2. * (sij[..., t, i, j] + sij[..., t, j, i])
                    # gij[..., i, j] += 1./2. * (sij[..., i, j] - sij[..., j, i]) #anti-symmetric part
                else:
                    eij[..., t, i, j] = eij[..., t, j, i]
                    # gij[..., i, j] = -gij[..., j, i] #anti-symmetric part

    gij = sij - eij
    return eij, gij


def get_jacobian_xyz_ijk(xx, yy, zz=None):
    """
    Returns diagonal elements of Jacobian between index space basis and physical space
    ... This returns xyz_orientations for get_duidxj_tensor().
    ... Further details can be found in the docstring of get_duidxj_tensor()

    Parameters
    ----------
    xx: 2d/3d array, a grid of x-coordinates
    yy: 2d/3d array, a grid of y-coordinates
    zz: 2d/3d array, a grid of z-coordinates

    Returns
    -------
    jacobian: 1d array
        ... expected icput of xyz_orientations for get_duidxj_tensor
    """

    dim = len(xx.shape)
    if dim == 2:
        x = xx[0, :]
        y = yy[:, 0]
        increments = [cp.nanmean(cp.diff(x)), cp.nanmean(cp.diff(y))]
    elif dim == 3:
        x = xx[0, :, 0]
        y = yy[:, 0, 0]
        z = zz[0, 0, :]
        increments = [cp.nanmean(cp.diff(x)), cp.nanmean(cp.diff(y)), cp.nanmean(cp.diff(z))]
    else:
        raise ValueError('... xx, yy, zz must have dimensions of 2 or 3. ')

    mapping = {True: 1, False: -1}
    jacobian = cp.asarray([mapping[increment.item() > 0] for increment in increments])  # Only diagonal elements
    return jacobian


def get_grid_spacing(xx, yy, zz=None):
    """Returns a grid spacing- the given grids must be evenly spaced"""
    dim = len(xx.shape)
    if dim == 2:
        dx = cp.abs(xx[0, 1] - xx[0, 0])
        dy = cp.abs(yy[1, 0] - yy[0, 0])
        return dx, dy
    elif dim == 3:
        dx = cp.abs(xx[0, 1, 0] - xx[0, 0, 0])
        dy = cp.abs(yy[1, 0, 0] - yy[0, 0, 0])
        dz = cp.abs(zz[0, 0, 1] - zz[0, 0, 0])
        return dx, dy, dz


def rankine_vortex_2d(xx, yy, x0=0, y0=0, gamma=1., a=1.):
    """
    Reutrns a 2D velocity field with a single Rankine vortex at (x0, y0)

    Parameters
    ----------
    xx: numpy array
        x-coordinate, 2d grid
    yy: numpy array
        y-coordinate, 2d grid
    x0: float
        x-coordinate of the position of the rankine vortex
    y0: float
        y-coordinate of the position of the rankine vortex
    gamma: float
        circulation of the rankine vortex
    a: float
        core radius of the rankine vortex

    Returns
    -------
    udata: (ux, uy)

    """
    rr, phi = cart2pol(xx - x0, yy - y0)

    cond = rr <= a
    ux, uy = cp.empty_like(rr), cp.empty_like(rr)
    # r <= a
    ux[cond] = -gamma * rr[cond] / (2 * cp.pi * a ** 2) * cp.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * cp.pi * a ** 2) * cp.cos(phi[cond])
    # r > a
    ux[~cond] = -gamma / (2 * cp.pi * rr[~cond]) * cp.sin(phi[~cond])
    uy[~cond] = gamma / (2 * cp.pi * rr[~cond]) * cp.cos(phi[~cond])

    udata = cp.stack((ux, uy))

    return udata


def cart2pol(x, y):
    """
    Transformation: Cartesian coord to polar coord

    Parameters
    ----------
    x: numpy array
    y: numpy array

    Returns
    -------
    r: numpy array
    phi: numpy array
    """
    r = cp.sqrt(x ** 2 + y ** 2)
    phi = cp.arctan2(y, x)

    return r, phi


# used in training loop
def compute_vorticity(velocity_field):
    # velocity_field has shape (batch_size, 2, 64, 64)
    batch_size = velocity_field.shape[0].cpu()
    num_rows = velocity_field.shape[2].cpu()
    num_cols = velocity_field.shape[3].cpu()
    x, y = list(range(num_cols)), list(range(num_rows))
    xx, yy = cp.meshgrid(x, y)

    all_vorticity = cp.zeros((velocity_field.shape[0],
                                1,
                                velocity_field.shape[2],
                                velocity_field.shape[3]))

    for i in range(batch_size):
        # curl function takes (dim, num_rows, num_cols)
        udata = velocity_field[i]
        omega = curl(udata, xx=xx, yy=yy)
        print(omega.shape)
        exit()
        all_vorticity[i] = omega

    return all_vorticity




