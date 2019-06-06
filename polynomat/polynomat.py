import numpy as np


class PolynoMat:
    """"""
    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other


def SBR2(R, delta, maxiter):
    R0 = R.copy()
    g = 1+delta
    iter = 0
    Hp = np.zeros_like(R, dtype='complex128')
    Hp[:, :, 0] = np.eye(R.shape[0])
    gs = []
    r2s = []
    N4 = np.linalg.norm(R)**2
    while g > delta and iter < maxiter:
        # 1) Find max off-diagonal element
        j, k, tau, g = off_diag_search(R)
        print("j, k, tau, g, ||R||^2: ", j, k, tau, g, np.linalg.norm(R)**2)
        gs.append(g)
        r2s.append(np.linalg.norm(R)**2)
        if g > delta:
            iter += 1
            B, Btilde = delay_matrix(k, tau, p=R.shape[0], max_tau=(R.shape[-1]-1)//2)
            # Btilde, B = _delay_matrix_(k, tau, p=R.shape[0], max_tau=(R.shape[-1] - 1) // 2)
            Rpt = fftpmm(B, R)
            Rp = fftpmm(Rpt, Btilde)
            Hp = fftpmm(B, Hp)
            Q, QH = rotation_matrix(j, k, Rp)
            Rt = fftpmm(Q, Rp)
            Rp = fftpmm(Rt, QH)
            Hp = fftpmm(Q, Hp)
            R = trim(Rp, 0.99, N4)
            # Hp = trim(Hp, 0.99, N4)
    return Hp, gs, r2s

def MS_SBR2(R, delta, maxiter):
    N4 = np.linalg.norm(R)**2
    g = 1+delta
    iter = 0
    Hp = np.zeros_like(R, dtype='complex128')
    Hp[:, :, 0] = np.eye(R.shape[0])
    gs = []
    r2s = []
    Rp = R
    while iter < maxiter and g > delta:
        # Locate and shift max off-diagonal elements
        Li = 0
        blacklist = []
        prs = []
        Rps = []
        while g > delta and len(blacklist) < R.shape[0]-1:
            j, k, tau, g = off_diag_search(R, blacklist=blacklist)
            print("j, k, tau, g, ||R||^2: ", j, k, tau, g, np.linalg.norm(R)**2)
            prs.append((j, k))
            blacklist.append(j)
            blacklist.append(k)
            if g > delta:
                P, Ph = delay_matrix(k, tau, p=R.shape[0], max_tau=(R.shape[-1]-1)//2)
                Rp = fftpmm(P, fftpmm(Rp, Ph))
                Hp = fftpmm(P, Hp)
                Li += 1
        # Perform a sequence of Jacobi rotations
        for (j, k) in prs:
            Q, Qh = rotation_matrix(j, k, Rp)
            Rp = fftpmm(Q, fftpmm(Rp, Qh))
            Hp = fftpmm(Q, Hp)
        iter += 1
        print(iter)
        r2s.append(np.linalg.norm(Rp)**2)
        Rp = trim(Rp, 0.99, N4)
    return Hp, gs, r2s

def trim(R, mu, N4):
    max_tau = (R.shape[-1]-1)//2
    tod = -1
    for idx in np.arange(max_tau):
        D = R[:, :, max_tau-idx:max_tau+idx+2]
        if np.linalg.norm(D)**2 <= (1-mu)*N4:
            tod += 1
        else:
            break
    if tod >= 0:
        R[:, :, max_tau-tod:max_tau+tod+2] = np.zeros((R.shape[0], R.shape[1], (tod+1)*2))
    return R

def paraconjugate(X):
    Xtmp = X[:, :, 1:]
    Xtmp = np.flip(Xtmp, -1)
    Xtmp = np.concatenate((X[:, :, 0:1], Xtmp), -1)
    Xtmp = np.conj(np.transpose(Xtmp, (1, 0, 2)))
    return Xtmp

def off_diag_search(X, blacklist=()):
    # 1) Find dominant off-diagonal element
    maxtau = (X.shape[-1]-1)//2
    coords = None
    Xcoords = 0
    for j in [x for x in np.arange(X.shape[0]) if x not in blacklist]:
        for k in [y for y in np.arange(j+1, X.shape[1]) if y not in blacklist]:
            for t in np.arange(X.shape[-1]):
                if np.abs(X[j, k, t]) > Xcoords:
                    coords = [j, k, t]
                    Xcoords = np.abs(X[j, k, t])
    if coords[-1] > maxtau:
        coords[-1] = coords[-1]-X.shape[-1]
    return coords[0], coords[1], coords[2], Xcoords

def delay_matrix(k, t, p, max_tau):
    """ Create an elementary delay polynomial matrix

    Arguments:
            k (int): The row/column number  to apply the delay
            t (int): The number of units time to delay by
            p (int): The dimension of the (square) pxp lag-zero matrix
            max_tau (int): The numbers of signals to locate
    """
    assert np.abs(t) <= max_tau, "Delay time too long for depth of array"
    B = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    B[:, :, 0] = np.eye(p, p)
    B[k, k, 0] = 0
    B[k, k, t] = 1
    Btilde = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    Btilde[:, :, 0] = np.eye(p, p)
    Btilde[k, k, 0] = 0
    Btilde[k, k, -t] = 1
    return B, Btilde

def rotation_matrix(j, k, X):
    p = X.shape[0]
    max_tau = (X.shape[-1]-1)//2
    phi = np.angle(X[j, k, 0])

    cs = np.cos(np.arctan2((2*np.abs(X[j, k, 0])), (X[j, j, 0].real-X[k, k, 0].real))/2)
    ss = np.sin(np.arctan2((2*np.abs(X[j, k, 0])), (X[j, j, 0].real-X[k, k, 0].real))/2)

    Q = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    Q[:, :, 0] = np.eye(p)
    Q[j, j, 0] = cs
    Q[j, k, 0] = ss*np.exp(1j*phi)
    Q[k, j, 0] = -ss*np.exp(-1j*phi)
    Q[k, k, 0] = cs
    # print("Q: ",Q[:,:,0])
    QH = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    QH[:, :, 0] = np.eye(p)
    QH[j, j, 0] = cs
    QH[j, k, 0] = -ss*np.exp(1j*phi)
    QH[k, j, 0] = ss*np.exp(-1j*phi)
    QH[k, k, 0] = cs
    return Q, QH

def fftpmm(H, R):
    """Algorithm 1 taken from Redif, S., & Kasap, S. (2015). Novel Reconfigurable Hardware Architecture for Polynomial Matrix Multiplications. Tvlsi, 23(3), 454–465."""
    (p, _, N) = H.shape
    Hfft = np.apply_along_axis(fft_pack.fft, -1, H, n=N)
    Rfft = np.apply_along_axis(fft_pack.fft, -1, R, n=N)
    reHfft = Hfft.real
    imHfft = Hfft.imag
    reRfft = Rfft.real
    imRfft = Rfft.imag
    reCfft = np.zeros((p, p, N))
    imCfft = np.zeros((p, p, N))

    for ix in np.arange(p):
        for jx in np.arange(p):
            for kx in np.arange(2*p):
                for t in np.arange(N):
                    if kx < p:
                        reCfft[ix, jx, t] += reHfft[ix, kx, t]*reRfft[kx, jx, t]
                        imCfft[ix, jx, t] += reHfft[ix, kx, t]*imRfft[kx, jx, t]
                    else:
                        reCfft[ix, jx, t] -= imHfft[ix, kx-p, t]*imRfft[kx-p, jx, t]
                        imCfft[ix, jx, t] += imHfft[ix, kx-p, t]*reRfft[kx-p, jx, t]
    # for ix in np.arange(p):
    #     for jx in np.arange(p):
    #         for kx in np.arange(2*p):
    #             for t in np.arange(N):
    #                 if kx < p:
    #                     imCfft[ix, jx, t] += reHfft[ix, kx, t]*imRfft[kx, jx, t]
    #                 else:
    #                     imCfft[ix, jx, t] += imHfft[ix, kx-p, t]*reRfft[kx-p, jx, t]
    Cfft = reCfft+1.0j*imCfft
    C = np.apply_along_axis(fft_pack.ifft, -1, Cfft, n=N)
    # plt.plot(abs(C.flatten()))
    # plt.show()
    # C.real[abs(C.real) < 1.0e-8] = 0
    # C.imag[abs(C.imag) < 1.0e-8] = 0
    # plt.plot(abs(C.flatten()))
    # plt.show()
    return C

def _polyconvmat_(F, G):

    return


def print_polynomial(z):
    prst = ""
    maxt = (len(z)-1)//2
    for pos in np.arange(maxt):
        if z[pos]!=0:
            if z[pos]==1:
                prst += "z^{} + ".format(pos)
            else:
                prst += "{}z^{} + ".format(z[pos], pos)
    if z[maxt-1]!=0:
        if z[maxt-1]==1:
            prst += "z + "
        else:
            prst += "{}z + ".format(z[maxt-1])
    if z[maxt]!=0:
        prst += "{} + ".format(z[maxt])
    for pos in np.arange(maxt+1, len(z)):
        if z[pos]!=0:
            if z[pos]==1:
                prst += "z^-{} + ".format(pos)
            else:
                prst += "{}z^{} + ".format(z[pos], maxt-pos)

    print(prst[:-2])

def polynom_steervec(mics, theta, max_tau=100, sound_speed=343.1):
    incidentdir = np.array([-np.cos(theta), -np.sin(theta)])
    incidentdir.shape = (2, 1)
    tau = np.dot(mics, incidentdir)/sound_speed
    Az = np.sinc(np.tile(np.arange(-max_tau, max_tau+1), (4, 1))-tau)
    Az *= np.blackman(Az.shape[-1])
    Az = np.flip(np.roll(Az, max_tau), -1)
    return Az

