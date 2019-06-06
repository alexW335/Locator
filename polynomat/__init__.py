import numpy as np
import pickle
import scipy.signal
import numba
from tqdm import tqdm


class PolynoMat(np.ndarray):

    def __new__(cls, input_array, max_tau=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = np.asarray(input_array).view(cls)
        # set the new 'info' attribute to the value passed
        obj.max_tau = (obj.shape[-1]-1)//2 if obj.shape[-1] != 1 else 0

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.max_tau = getattr(obj, 'max_tau', (self.shape[-1]-1)//2 if self.shape[-1] != 1 else 0)

    def __str__(self):
        R = PolynoMat(self.copy().round(4))
        strep = ""
        t = []
        mlen = 0
        if len(R.shape) > 1:
            for r in np.arange(R.shape[0]):
                tt = []
                for c in np.arange(R.shape[1]):
                    y = R.print_polynomial((r, c), ret=True)
                    tt.append(y)
                    if len(y) > mlen:
                        mlen = len(y)
                t.append(tt)
            strep += u"\u2308"
            for r in np.arange(R.shape[0]):
                if r != R.shape[0] - 1 and r != 0:
                    strep += "|"
                if r == R.shape[0] - 1:
                    strep += u"\u230A"
                for c in np.arange(R.shape[1]):
                    strep += "{:^{ln}}".format(t[r][c], ln=mlen + 2)
                if r != R.shape[0] - 1 and r != 0:
                    strep += "|"
                if r == 0:
                    strep += u"\u2309"
                if r == R.shape[0] - 1:
                    strep += u"\u230B"
                strep += "\n"
        else:
            strep += R.print_polynomial((0, 0), single=True, ret=True)
        return strep

    def H(self):
        X = self.copy()
        Xtmp = np.flip(X, -1)
        Xtmp = np.conj(np.transpose(Xtmp, (1, 0, 2)))
        return Xtmp

    def h(self):
        return self.H()

    def print_polynomial(self, coords:tuple, single=False, ret=False):
        if single:
            z = self[:]
        else:
            z = self[coords[0], coords[1], :]

        prst = ""
        maxt = (len(z) - 1) // 2 if len(z) != 1 else 0

        for pos in np.arange(maxt - 1):
            if z[pos] != 0:
                if z[pos] == 1:
                    prst += "z^{} + ".format(maxt - pos)
                else:
                    if z[pos].real == 0:
                        prst += "{}jz^{} + ".format(z[pos].imag, maxt - pos)
                    if z[pos].imag == 0:
                        prst += "{}z^{} + ".format(z[pos].real, maxt - pos)
                    if z[pos].imag != 0 and z[pos].real != 0:
                        prst += "{}z^{} + ".format(z[pos], maxt - pos)

        if z[maxt - 1] != 0 and maxt != 0:
            if z[maxt - 1] == 1:
                prst += "z + "
            else:
                if z[maxt - 1].real == 0:
                    prst += "{}jz + ".format(z[maxt - 1].imag)
                if z[maxt - 1].imag == 0:
                    prst += "{}z + ".format(z[maxt - 1].real)
                if z[maxt - 1].imag != 0 and z[maxt - 1].real != 0:
                    prst += "{}z + ".format(z[maxt - 1])

        if z[maxt] != 0:
            if z[maxt].real == 0:
                prst += "{}j + ".format(z[maxt].imag)
            if z[maxt].imag == 0:
                prst += "{} + ".format(z[maxt].real)
            if z[maxt].imag != 0 and z[maxt].real != 0:
                prst += "{} + ".format(z[maxt])

        for pos in np.arange(maxt + 1, len(z)):
            if z[pos] != 0:
                if z[pos] == 1:
                    prst += "z^{} + ".format(maxt - pos)
                else:
                    if z[pos].real == 0:
                        prst += "{}jz^{} + ".format(z[pos].imag, maxt - pos)
                    if z[pos].imag == 0:
                        prst += "{}z^{} + ".format(z[pos].real, maxt - pos)
                    if z[pos].real != 0 and z[pos].imag != 0:
                        prst += "{}z^{} + ".format(z[pos], maxt - pos)

        if prst[:-2] == '':
            if ret:
                return "0"
            else:
                print("0")
        else:
            if ret:
                return prst[:-3]
            else:
                print(prst[:-3])


def pmm(A, B, truncate=True):
    assert A.shape[1] == B.shape[0], "Matrix dimensions don't match."
    max_tau = min((A.shape[-1] - 1) // 2, (B.shape[-1] - 1) // 2)
    if truncate:
        C = np.zeros((A.shape[0], B.shape[1], min(A.shape[-1], B.shape[-1])), dtype='complex128')
        halfC = ((A.shape[-1]+B.shape[-1]-1)-1)//2
    else:
        C = np.zeros((A.shape[0], B.shape[1], A.shape[-1]+B.shape[-1]-1), dtype='complex128')

    for jx in np.arange(B.shape[1]):
        for ix in np.arange(A.shape[0]):
            for p in np.arange(B.shape[0]):
                if truncate:
                    t = scipy.signal.fftconvolve(B[p,jx,:], A[ix,p,:])[halfC-max_tau:halfC+max_tau+1]
                else:
                    t = scipy.signal.fftconvolve(B[p,jx,:], A[ix,p,:])
                C[ix, jx, :] += t
    return PolynoMat(C)


# @numba.jit(nopython=True)
# def trim(Hp:np.array, epsilon=1.0e-6):
#     max_tau = (Hp.shape[-1] - 1) // 2
#     absHp = np.abs(Hp)
#     absHpsum = np.sum(absHp)
#     mu = epsilon * absHpsum
#     cs = np.sum(absHp, axis=(0, 1))
#     cs = cs.flatten()
#     t = np.zeros((cs.shape[0] // 2 + 1, 2))
#     t[:, 0] = cs[:max_tau + 1]
#     t[:-1, 1] = cs[(max_tau * 2 + 1):max_tau:-1]
#     t = np.sum(t, axis=1)
#     cs = np.cumsum(t)
#     msk = np.where(cs > mu, True, False)
#     msklen = np.sum(msk)
#     msk = np.tile(np.concatenate((msk[:-1], msk[::-1])), (Hp.shape[0] * Hp.shape[1]))
#     Hpr = Hp.ravel()[msk]
#     return Hpr.reshape((Hp.shape[0], Hp.shape[0], 2 * msklen - 1))

# @numba.jit
def trim(Hp:np.array, epsilon=1.0e-6):
    max_tau = (Hp.shape[-1]-1)//2
    absHp = np.abs(Hp)
    mu = epsilon*np.sum(absHp)
    cur = 1
    runnorm = np.sum(absHp[..., 0])
    runnorm += np.sum(absHp[..., -1])
    while runnorm <= mu and cur <= max_tau-1:
        runnorm += np.sum(absHp[..., cur])
        runnorm += np.sum(absHp[..., -(cur + 1)])
        cur += 1
    if cur == max_tau and runnorm < mu:
        return Hp[..., max_tau:max_tau + 1]
    elif cur == 1:
        return Hp
    else:
        return Hp[..., cur-1:-(cur-1)]


def SBR2(R, delta, maxiter, loss):
    Rp = PolynoMat(R.copy())
    # R0 = Rp.copy()
    # g = 1 + delta
    Hp = PolynoMat(np.zeros_like(Rp, dtype='complex128'))
    Hp[:, :, Hp.max_tau] = np.eye(Rp.shape[0], dtype='complex128')
    gs = []
    r2s = []
    ls = []
    for _ in tqdm(np.arange(maxiter)):
        (j, k, tau), g = off_diag_search(Rp, norm=np.inf)
        gs.append(g)
        r2s.append(np.sum(np.abs(Rp)))
        if g > delta:
            if tau != 0:
                # Rp = PolynoMat(np.pad(Rp, ((0, 0), (0, 0), (np.abs(tau), np.abs(tau))), mode='constant', constant_values=0))
                Rp = PolynoMat(np.dstack((np.zeros((Rp.shape[0],Rp.shape[1],np.abs(tau))), Rp,
                                          np.zeros((Rp.shape[0],Rp.shape[1],np.abs(tau))))))
                Rp[:, k, :] = np.roll(Rp[:, k, :], tau, axis=-1)
                Rp[k, :, :] = np.roll(Rp[k, :, :], -tau, axis=-1)
                # Hp = PolynoMat(np.pad(Hp, ((0, 0), (0, 0), (np.abs(tau), np.abs(tau))), mode='constant', constant_values=0))
                Hp = PolynoMat(np.dstack((np.zeros((Hp.shape[0], Hp.shape[1], np.abs(tau))), Hp,
                                          np.zeros((Hp.shape[0], Hp.shape[1], np.abs(tau))))))
                Hp[k, :, :] = np.roll(Hp[k, :, :], -tau, axis=-1)
                # Hp = pmm(B, Hp, truncate=False)

            # Trim off zeros but nothing else
            # Hp = trim(Hp, 1)
            # Rp = trim(Rp, 1)

            Q, QH = rotation_submatrix2D(j, k, Rp)

            # Only operate on the appropriate rows and columns
            Rp[[j, k], :, :] = np.einsum('ij,jkl->ikl', Q, Rp[[j, k], :, :])
            Rp[:, [j, k], :] = np.einsum('jil,ik->jkl', Rp[:, [j, k], :], QH)

            Hp[[j, k], :, :] = np.einsum('ij,jkl->ikl', Q, Hp[[j, k], :, :])

            # Trim the result
            Hp = PolynoMat(trim(np.array(Hp), loss))
            # print("before ", Rp.shape)
            Rp = PolynoMat(trim(np.array(Rp), loss))
            # print("after ", Rp.shape)
            ls.append(Rp.shape[-1])
        else:
            break
    return Hp, gs, r2s, Rp, ls


def off_diag_search(X:PolynoMat, norm=np.inf):
    """ Find the largest magnitude off-diagonal element in parahermitian polynomial input matrix X

    Arguments:
            X (PolynoMat): The parahermitian polynomial input matrix
    """

    coords = None
    high = 0

    if norm == np.inf:
        Xcp = PolynoMat(X.copy())
        Xcp[np.arange(Xcp.shape[0]), np.arange(Xcp.shape[1])] = 0.0
        coords = list(np.unravel_index(np.argmax(np.abs(np.array(Xcp))), np.array(Xcp).shape))
        high = np.abs(Xcp[coords[0], coords[1], coords[2]])

    elif norm == 2:
        Xcp = PolynoMat(np.power(X.copy(), 2))
        for tauindx in np.arange(X.shape[-1]):
            np.fill_diagonal(Xcp[:, :, tauindx], 0.0)
            for rindx in np.arange(X.shape[0]):
                n = np.sum(Xcp[rindx, :, tauindx])
                if n > high:
                    high = n
                    coords = [rindx, tauindx]
    else:
        return None

    coords[-1] = X.max_tau - coords[-1]

    if norm == np.inf:
        if coords[1] < coords[0]:
            t = coords[0:2]
            coords[0:2] = t[::-1]
            coords[-1] *= -1
        return tuple(coords), high
    else:
        return tuple(coords), np.sqrt(high)


def delay_matrix(k, t, X:PolynoMat):
    """ Create an elementary delay polynomial matrix

    Arguments:
            k (int): The row/column index to apply the delay
            t (int): The number of units time to delay by
            X (int): A template polynomial matrix for getting dimensions
    """
    p = X.shape[0]
    mt = X.max_tau
    assert np.abs(t) <= mt, "Delay time too long for depth of array"
    D = PolynoMat(np.zeros_like(X, dtype='complex128'))
    D[:, :, mt] = np.eye(p, p)
    D[k, k, mt] = 0
    D[k, k, mt-t] = 1
    Dh = PolynoMat(np.zeros_like(X, dtype='complex128'))
    Dh[:, :, mt] = np.eye(p, p)
    Dh[k, k, mt] = 0
    Dh[k, k, mt+t] = 1
    return D, Dh


def rotation_submatrix2D(j, k, X: PolynoMat):
    mt = X.max_tau
    phi = np.angle(X[j, k, mt])
    assert j < k, "Entry must be in top-right half of matrix"

    cs = np.cos(np.arctan2((2 * np.abs(X[j, k, mt])), (X[j, j, mt].real - X[k, k, mt].real)) / 2)
    ss = np.sin(np.arctan2((2 * np.abs(X[j, k, mt])), (X[j, j, mt].real - X[k, k, mt].real)) / 2)

    Q = np.eye(2, dtype='complex128')
    Q[0, 0] = cs
    Q[0, 1] = ss * np.exp(1j * phi)
    Q[1, 0] = -ss * np.exp(-1j * phi)
    Q[1, 1] = cs

    QH = np.eye(2, dtype='complex128')
    QH[0, 0] = cs
    QH[0, 1] = -ss * np.exp(1j * phi)
    QH[1, 0] = ss * np.exp(-1j * phi)
    QH[1, 1] = cs
    return Q, QH

def space_time_covm(data, max_tau):
    """ From PEVDToolbox/SpaceTimeCovMat.m from PEVD
    toolbox by S Weiss.
    """
    print("Generating space-time covariance matrix")
    M = data.shape[1]
    L = data.shape[0]
    R = PolynoMat(np.zeros((M, M, 2*max_tau+1), dtype='complex128'))
    Lm = L-2*max_tau

    Xref = data[max_tau:max_tau+Lm, :]
    for tau in tqdm(np.arange(0, 2*max_tau+1)):
        Xshift = data[tau:tau+Lm, :]
        R[:, :, 2*max_tau-tau] = np.dot(Xref.T, Xshift)/Lm
    R = (R+R.H())/2
    return R


if __name__ == "__main__":
    print("hello there, you weren't really meant to run this file")
