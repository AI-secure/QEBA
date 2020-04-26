import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from disk_mat import DiskMatrix

def mult(A, B):
    if isinstance(A, DiskMatrix):
        assert isinstance(B, np.ndarray)
        return A.rightdot(B)
    elif isinstance(B, DiskMatrix):
        return B.leftdot(A)
    else:
        return A.dot(B)

def gen_topK_colspace(A, k, n_iter=1):
    # Input
    # A - an (m*n) matrix
    # k - rank
    # n_iter - numer of normalized power iterations
    # Output
    # Q: an (k*n) matrix 

    import time
    t_cur = time.time()
    (m, n) = A.shape

    if (True):
        #Q = np.random.uniform(low=-1.0, high=1.0, size=(k, m)).dot(A).T
        Q = mult(np.random.uniform(low=-1.0, high=1.0, size=(k, m)), A).T
        print (time.time() - t_cur)
        t_cur = time.time()
        Q, _ = lu(Q, permute_l=True)
        print (time.time() - t_cur)
        t_cur = time.time()
        for it in range(n_iter):
            #Q = A.dot(Q)
            Q = mult(A, Q)
            print (time.time() - t_cur)
            t_cur = time.time()
            Q, _ = lu(Q, permute_l=True)
            print (time.time() - t_cur)
            t_cur = time.time()
            #Q = Q.T.dot(A).T
            Q = mult(Q.T, A).T
            print (time.time() - t_cur)
            t_cur = time.time()
            if it + 1 < n_iter:
                (Q, _) = lu(Q, permute_l=True)
            else:
                (Q, _) = qr(Q, mode='economic')
            print (time.time() - t_cur)
            t_cur = time.time()
    else:
        raise NotImplementedError()
    print ("DONE")
    return Q.T


class PCAGenerator:
    # if the input dimension is too large (e.g., ImageNet), we should set approx=True to use randomized PCA
    def __init__(self, N_b, X_shape=None, batch_size=32, preprocess=None, approx=False, basis_only = False):
        self.N_b = N_b
        self.X_shape = X_shape
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.basis = None
        self.approx = approx
        self.basis_only = basis_only

    def fit(self, X):
        if self.X_shape is None:
            raise RuntimeError("X_shape must be passed")
        assert len(X.shape) == 2
        N = X.shape[0]
        if self.approx:
            print ("Using approx pca")
            #import fbpca
            #U, S, Vt = fbpca.pca(A=X, k=self.N_b, raw=True)
            Vt = gen_topK_colspace(A=X, k=self.N_b)
            self.basis = Vt
        else:
            from sklearn.decomposition import PCA
            model = PCA(self.N_b)
            model.fit(X)
            self.basis = model.components_
        if self.basis_only:
            self.basis = self.basis.reshape(self.basis.shape[0], *self.X_shape)

    def save(self, path):
        np.save(path, self.basis.reshape(self.N_b, *self.X_shape))

    def load(self, path):
        self.basis = np.load(path)
        self.X_shape = self.basis.shape[1:]
        if self.basis_only:
            self.basis = self.basis
        else:
            self.basis = self.basis.reshape(self.basis.shape[0], -1)

    def generate_ps(self, inp, N, level=None):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
        
        if self.basis is None:
            raise RuntimeError("Must fit or load the model first")

        #ps = []
        #for _ in range(N):
        #    #rv = np.random.randn(self.N_b, 1,1,1)
        #    #pi = (rv * self.basis).sum(axis=0)
        #    #ps.append(pi)
        #    rv = np.random.randn(1, self.N_b)
        #    pi = rv @ self.basis
        #    ps.append(pi)
        ##ps = np.stack(ps, axis=0)
        #ps = np.concatenate(ps, axis=0).reshape(N, *self.X_shape)
        import time
        if self.basis_only:
            rv = np.random.randint(self.N_b, size=(N,))
            ps = self.basis[rv]
        else:
            rv = np.random.randn(N, self.N_b)
            ps = rv.dot(self.basis).reshape(N, *self.X_shape)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps

    def calc_rho(self, gt, inp, factor=4.0):
        all_cos2 = 0.0
        for vi in self.basis:
            #cosi = (vi*gt).sum() / np.sqrt( (vi**2).sum() * (gt**2).sum() )
            cosi = (vi.reshape(*self.X_shape)*gt).sum() / np.sqrt( (vi**2).sum() * (gt**2).sum() )
            all_cos2 += (cosi ** 2)
        rho = np.sqrt(all_cos2)
        return rho
