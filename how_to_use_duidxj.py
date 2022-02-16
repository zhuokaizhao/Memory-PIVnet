import numpy as np
import matplotlib.pyplot as plt

#### HELPER FUNCTIONS ####
######## Coordinate transformation ########
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
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return r, phi


def fix_udata_shape(udata):
    """
    Reshapes udata into a standard format (dim, height, width, (depth), duration).
    ... udata.shape changes from (dim, height, width, (depth)) to (dim, height, width, (depth), 1)

    Parameters
    ----------
    udata: nd array,
          ... with shape (height, width, depth) (3D) or  (height, width, duration) (2D)
          ... OR shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    Returns
    -------
    udata: nd array, with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
            return udata
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))
            return np.stack((ux, uy))

    elif shape[0] == 3:
        dim = 3
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            nrows, ncols, nstacks, duration = ux.shape
            return udata
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
            return np.stack((ux, uy, uz))


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
        ... expected input of xyz_orientations for get_duidxj_tensor
    """

    dim = len(xx.shape)
    if dim == 2:
        x = xx[0, :]
        y = yy[:, 0]
        increments = [np.nanmean(np.diff(x)), np.nanmean(np.diff(y))]
    elif dim == 3:
        x = xx[0, :, 0]
        y = yy[:, 0, 0]
        z = zz[0, 0, :]
        increments = [np.nanmean(np.diff(x)), np.nanmean(np.diff(y)), np.nanmean(np.diff(z))]
    else:
        raise ValueError('... xx, yy, zz must have dimensions of 2 or 3. ')

    mapping = {True: 1, False: -1}
    jacobian = np.asarray([mapping[increment > 0] for increment in increments])  # Only diagonal elements
    return jacobian


def get_grid_spacing(xx, yy, zz=None):
    """
    Returns spacings (dx, dy, dz) along the x-, y-, and z- direction

    Parameters
    ----------
    xx: 2d/3d array, x-coordinate of the position
        ... xx, yy = np.meshgrid(x, y) with x = np.linspace(xmin, xmax, n)
        ... xx[0, :] must be equal to x.
    yy: 2d/3d array, y-coordinate of the position
        ... yy[:, 0] must be equal to y.
    zz: 3d array, z-coordinate of the position (optional)
        ... xx, yy, zz = np.meshgrid(x, y, z)
        ... zz[0, 0, :] must be equal to z.

    Returns
    -------
    dx, dy, dz(optional): tuple of float, spacings of the positional grids

    """
    dim = len(xx.shape)
    if dim == 2:
        dx = np.abs(xx[0, 1] - xx[0, 0])
        dy = np.abs(yy[1, 0] - yy[0, 0])
        return dx, dy
    elif dim == 3:
        dx = np.abs(xx[0, 1, 0] - xx[0, 0, 0])
        dy = np.abs(yy[1, 0, 0] - yy[0, 0, 0])
        dz = np.abs(zz[0, 0, 1] - zz[0, 0, 0])
        return dx, dy, dz


### RATE OF STRAIN TENSOR ###
def get_duidxj_tensor(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]),
                      xx=None, yy=None, zz=None):
    """
    Assumes udata has a shape (d, nrows, ncols, duration) or  (d, nrows, ncols)
    ... one can easily make udata by np.stack((ux, uy))

    Important Warning:
    ... udata is np.stack((ux, uy, uz))
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

        duxdx = np.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = np.gradient(ux, dy, axis=0) * xyz_orientations[
            1]  # +dy is the column up. np gradient computes difference by going DOWN in the column, which is the opposite
        duydx = np.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = np.gradient(uy, dy, axis=0) * xyz_orientations[1]
        sij = np.zeros((nrows, ncols, duration, dim, dim))
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
        duxdx = np.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = np.gradient(ux, dy, axis=0) * xyz_orientations[1]
        duxdz = np.gradient(ux, dz, axis=2) * xyz_orientations[2]
        duydx = np.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = np.gradient(uy, dy, axis=0) * xyz_orientations[1]
        duydz = np.gradient(uy, dz, axis=2) * xyz_orientations[2]
        duzdx = np.gradient(uz, dx, axis=1) * xyz_orientations[0]
        duzdy = np.gradient(uz, dy, axis=0) * xyz_orientations[1]
        duzdz = np.gradient(uz, dz, axis=2) * xyz_orientations[2]

        sij = np.zeros((nrows, ncols, nstacks, duration, dim, dim))
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


### TENSOR DECOMPOSITION ###
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

    eij = np.zeros(sij.shape)
    # gij = np.zeros(sij.shape) #anti-symmetric part
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


### VECTOR OPERATION ###
def curl(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]),
         xx=None, yy=None, zz=None, verbose=False):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
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
        omega = np.stack((omega1, omega2, omega3))
    else:
        print('Not implemented yet!')
        return None
    return omega


def div(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]), xx=None, yy=None, zz=None):
    """
    Computes divergence of a velocity field

    Parameters
    ----------
    udata: numpy array
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    div_u: numpy array
          ... div_u has a shape (height, width, depth, duration) (3D) or (height, width, duration) (2D)
    """
    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz, xyz_orientations=xyz_orientations, xx=xx, yy=yy,
                            zz=zz)  # shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    dim = len(sij.shape) - 3  # spatial dim
    div_u = np.zeros(sij.shape[:-2])
    for d in range(dim):
        div_u += sij[..., d, d]

    return div_u


# Sample velocity field
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
    ux, uy = np.empty_like(rr), np.empty_like(rr)
    # r <= a
    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    # r > a
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    udata = np.stack((ux, uy))

    return udata

def main():
    # Define a grid
    nx, ny = 30, 40
    x, y = np.linspace(-1, 1, 30), np.linspace(-1, 1, 40)
    xx, yy = np.meshgrid(x, y)
    # Get a sample velocity field (2D)
    udata = rankine_vortex_2d(xx, yy, a=0.5)
    print('Shape of udata is (dim, ny, nx):', udata.shape)

    # Get a rate-of-strain tensor, dui/dxj
    duidxj = get_duidxj_tensor(udata, xx=xx, yy=yy)
    print('Shape of duidxj is (ny, nx, duration, dim, dim):', duidxj.shape)

    # Decompose duidxj into symmetric and anti-symmetric parts
    eij, gij = decompose_duidxj(duidxj)
    print('\n#####Decomposition: duidxj = (symmetric tensor) + (anti-symmetric tensor) = eij + gij#####')
    print('\nNotice that eij and gij have the same shape as duidxj')
    print('Shape of eij is (ny, nx, duration, dim, dim):', eij.shape)
    print('Shape of gij is (ny, nx, duration, dim, dim):', gij.shape)

    ## HOW TO USE:
    print('\n#####How to use the duidxj array#####')
    print('duidxj, eij, and gij stores their values in the shape of (y, x, time(frame), tensor index i, tensor index j):')

    print('dux/dx = duidxj[:, :, :, 0, 0]')
    print('dux/dy = duidxj[:, :, :, 0, 1]')
    print('duy/dx = duidxj[:, :, :, 1, 0]')
    print('duy/dy = duidxj[:, :, :, 1, 1]')
    ##############################
    # Divergence
    divU = duidxj[..., 0, 0] + duidxj[..., 1, 1]
    # Curl
    curlU = duidxj[..., 1, 0] - duidxj[..., 0, 1]
    # Shear 1
    shear1U = duidxj[..., 0, 0] - duidxj[..., 1, 1]
    # Shear 2
    shear2U = duidxj[..., 1, 0] + duidxj[..., 0, 1]
    ##############################

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.streamplot(xx, yy, udata[0, ...], udata[1, ...], )

    vmin, vmax = -1.5, 1.5
    fig2 = plt.figure(2, figsize=(12, 3))
    axes2 = fig2.subplots(1, 4)
    cc0 = axes2[0].pcolormesh(xx, yy, divU[..., 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes2[1].pcolormesh(xx, yy, curlU[..., 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes2[2].pcolormesh(xx, yy, shear1U[..., 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes2[3].pcolormesh(xx, yy, shear2U[..., 0], cmap='coolwarm', vmin=vmin, vmax=vmax)

    titles = ['Dilation', 'Rotation (vorticity)', 'Shear 1', 'Shear 2']
    for ax, title in zip(axes2, titles):
        ax.set_title(title)

    # fig2.colorbar(cc0)
    fig2.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
