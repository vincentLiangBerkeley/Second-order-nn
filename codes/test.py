import unittest
from factorize import *
from naive_pcg import *
from tri_solve import *
from nn_util import *
import util, time

class TestFactorization(unittest.TestCase):
    def setUp(self):
        self.EPS = 1e-12
        self.l = 2
        self.dk = 2
        self.dx = self.dk*self.l

    def test_square(self):
        NN = [np.random.randn(self.dx,self.dk) for i in range(self.l)]
        A = [np.random.randn(self.dk,self.dk) for i in range(self.l)]

        R,S = QR_of_J(NN,A)
        RR = assemble_R(R,S)
        J = assemble_J(NN,A, self.l)

        u,s,v = np.linalg.svd(RR)
        u,s2,v = np.linalg.svd(J)
        self.assertTrue(np.max(abs(s-s2)) < self.EPS)

    def test_rectangle(self):
        NN = [np.random.randn(self.dx,self.dk) for i in range(self.l)]
        A = [np.random.randn(self.dk,3), np.random.randn(self.dk,4)]

        R,S = QR_of_J(NN,A)
        RR = assemble_R(R,S)
        J = assemble_J(NN,A, self.l)

        u,s,v = np.linalg.svd(RR)
        u,s2,v = np.linalg.svd(J)
        self.assertTrue(np.max(abs(s-s2)) < self.EPS)

class TestNaivePCG(unittest.TestCase):
    """Test class to test direct PCG implementation"""
    def setUp(self):
        # test pcg
        # testing the effectiveness of the preconditioner, just for matrices, without reference to neural nets
        l = 3
        dk = np.array([40, 30, 10])
        dx = np.sum(dk)
        NN = [np.random.randn(dx,dk[i]) for i in range(l)]
        A = [np.random.randn(dk[l-1],dk[i]) for i in range(l)]

        self.R,self.S = QR_of_J(NN,A)
        self.J = assemble_J(NN,A, l)

        self.n = self.J.shape[1]
        self.b = np.random.randn(self.n,1)
        self.x = np.zeros((self.n,1))
    
    def test_naive(self):
        J = np.random.randn(100,100)
        b = np.random.randn(100,1)
        x = np.random.randn(100,1)
        q,r = np.linalg.qr(J)

        sol1,i1 = pcg(J, 0.1, r, b, x)
        print('with preconditioning, iterations = ',i1)
        sol2,i2 = pcg(J, 0.1, np.eye(100), b, x)
        print('with no preconditioning, iterations = ',i2)

    def run_pcg_lamb(self, lamb):
        print("Testing naive pcg with lambda = %.2f"%lamb)
        R_precon = assemble_R_precon(self.R,self.S,np.sqrt(lamb))
        sol, i = pcg(self.J, lamb, np.eye(self.n), self.b, self.x)
        print('with no preconditioning, iterations = ',i)
        sol, i = pcg(self.J, lamb, R_precon, self.b, self.x)
        print('wth preconditioning, iterations = ',i)

    def test_PCG_lamb(self):
        lambs = [0.01, 0.1, 1, 10]
        for lamb in lambs:
            self.run_pcg_lamb(lamb)

class TestTriSolve(unittest.TestCase):
    def setUp(self):
        l = 2
        dk = 2
        dx = dk*l
        NN = [np.random.randn(dx,dk) for i in range(l)]
        A = [np.random.randn(dk,3), np.random.randn(dk,4)]
        self.lamb = 10

        self.R,S = QR_of_J(NN,A)
        self.SS = append_zero_rows(S)

        self.R_precon = assemble_R_precon(self.R,S,self.lamb)

        self.b = np.random.randn(self.R_precon.shape[0],1)

    def test_back_solve(self):
        sol = backsolve_R_i(self.R,self.SS,self.lamb,self.b)
        sol_exact = sp.solve_triangular(self.R_precon,self.b)
        self.assertTrue(np.max(np.abs(sol-sol_exact)) < 1e-15)

    def test_forward_solve(self):
        sol2 = backsolve_Rtr_i(self.R,self.SS,self.lamb,self.b)
        sol2_exact = sp.solve_triangular(self.R_precon.T,self.b,lower=True)

        self.assertTrue(np.max(np.abs(sol2-sol2_exact))< 1e-15)

class TestMultiply(unittest.TestCase):
    """Testing implicit multiplication of J and J^T"""
    def setUp(self):
        self.EPS = 1e-12
        self.N_0 = np.random.randn(50, 10)
        self.W_1 = np.random.randn(300, 50)
        self.W_2 = np.random.randn(5, 300)
        self.W = [self.W_1, self.W_2]
        self.F = [sigmoid_gradient, tanh_gradient]
        self.F_forward = [sigmoid, tanh]
        # Initialize small perturbation
        self.X_1 = np.random.randn(self.W_1.shape[0], self.W_1.shape[1])*self.EPS
        self.X_2 = np.random.randn(self.W_2.shape[0], self.W_2.shape[1])*self.EPS
        self.N_old = forward_params(self.W, self.N_0, self.F_forward)
        self.Y = util.multiply_J(self.N_old, self.W, self.F, [self.X_1, self.X_2])
        self.Jtr = util.form_Jtr(self.N_old, self.W, self.F)

    def test_multiply_J(self):
        
        N_new = forward_params([self.W_1+self.X_1, self.W_2+self.X_2], self.N_0, self.F_forward)
        
        self.assertTrue(np.max(abs(N_new[-1]-self.N_old[-1]-self.Y)) < 2 * self.EPS)

    def test_form_Jtr(self):
        X_vec = np.vstack((np.matrix(self.X_1.flatten('F')).T, np.matrix(self.X_2.flatten('F')).T))
        direct = np.dot(self.Jtr.T, X_vec)

        #indirect = multiply_J(N_old, W, F, [X_1, X_2]).flatten('F')
        indirect = self.Y.flatten('F')
        self.assertTrue(np.max(np.abs(direct - np.matrix(indirect).T)) < 1e-15)

    def test_multiply_Jtr(self):
        direct = np.dot(self.Jtr, self.Y.flatten('F'))
        indirect = util.multiply_Jtr(self.N_old, self.W, self.F, self.Y)
        X1 = indirect[0].flatten('F')
        X2 = indirect[1].flatten('F')
        x = np.vstack((np.matrix(X1).T, np.matrix(X2).T))
        self.assertTrue(np.max(abs(direct-x.T)) < 1e-15)

    def test_conversion(self):
        test = mat_to_vec([self.X_1, self.X_2])
        mat = vec_to_mat(test, [self.X_1.shape, self.X_2.shape])
        self.assertTrue(np.max(abs(mat[0] - X_1)) < 1e-16) 
    
class TestPCG(unittest.TestCase):
    """Testing PCG with implicit multiplication methods."""
    def setUp(self):
        # testing of PCG
        self.dk = np.array([32,16,8])
        self.N, self.W, self.F, self.NN_id, self.A_id, self.R_id, self.SS_id, self.NN_rk1, self.A_rk1, self.R_rk1, self.SS_rk1, self.x, self.b = construct_2layer_nn(self.dk, [sigmoid, tanh], [sigmoid_gradient, tanh_gradient])
        # generate weights 
        self.lambs = [0.01, 0.1, 1]

    def run_pcg(self, lamb, R, SS):
        start = time.clock()
        sol, i = util.pcg_i(self.N, self.W, self.F, lamb, R, SS, self.b, self.x)
        end = time.clock() - start
        start = time.clock()
        print('with preconditioning, iterations = %d, running time = %s'%(i, end))
        sol2, i2 = util.pcg_i(self.N, self.W, self.F, lamb, R, SS, self.b, self.x, is_p = False)
        end = time.clock() - start
        print('with no preconditioning, iterations = %d, running time = %s'%(i2, end))

    def test_pcg(self):
        for lamb in self.lambs:
            print("\nTesting PCG with identity activation lamb = %.2f"%lamb)
            self.run_pcg(lamb, self.R_id, self.SS_id)

    def test_pcg_rk1(self):
        for lamb in self.lambs:
            print("\nTesting PCG with rank-one approximation lamb = %.2f"%lamb)
            self.run_pcg(lamb, self.R_rk1, self.SS_rk1)

class TestApproximation(unittest.TestCase):
    """Testing the approximation effectiveness of Qr and thresholding"""
    def setUp(self):
        lamb = 0.01
        dk = np.array([32,16, 8])
        N, W, F, NN_id, A_id, self.R, self.SS, NN_rk1, A_rk1, self.R_rk1, self.SS_rk1, x0, b = \
                construct_2layer_nn(dk)
        
        self.Jtr_full = util.form_Jtr(N, W, F)

        A = np.dot(self.Jtr_full, self.Jtr_full.T) + lamb*np.eye(self.Jtr_full.shape[0])

        R_full = assemble_R_precon(self.R, self.SS, np.sqrt(lamb))
        A2 = np.dot(R_full.T, R_full)

        R_rk1_full = assemble_R_precon(self.R_rk1, self.SS_rk1, np.sqrt(lamb))
        A3 = np.dot(R_rk1_full.T, R_rk1_full)

        u,self.s,v = np.linalg.svd(A)
        u,self.s2,v = np.linalg.svd(A2)
        u,self.s3,v = np.linalg.svd(A3)

        k = np.linalg.cond(A)
        k2 = np.linalg.cond(A2)
        k3 = np.linalg.cond(A3)

    def test_id(self):
        print("\nTesting approximation of R^TR and J^TJ+lamb*I with identity activation.")
        print("The largest 3 singular values of R^TR are: %.2f %.2f %.2f"%(self.s2[0], self.s2[1], self.s2[2]))
        print("The largest 3 singular values of J^TJ+lamb*I are: %.2f %.2f %.2f"%(self.s[0], self.s[1], self.s[2]))

    def test_rk1(self):
        print("\nTesting approximation of R^TR and J^TJ+lamb*I with rank-one approximation.")
        print("The largest 3 singular values of R^TR are: %.2f %.2f %.2f"%(self.s3[0], self.s3[1], self.s3[2]))
        print("The largest 3 singular values of J^TJ+lamb*I are: %.2f %.2f %.2f"%(self.s[0], self.s[1], self.s[2]))

    def test_unthresh(self):
        print("\nTesting approximation of R^TR and J^TJ with unthresholding approximation.")
        R_rk1 = assemble_R_precon(self.R_rk1, self.SS_rk1, 0)
        R = assemble_R_precon(self.R, self.SS, 0)
        u, s, v = np.linalg.svd(self.Jtr_full.T)
        u, s1, v = np.linalg.svd(R_rk1)       
        u, s2, v = np.linalg.svd(R)

        print("The largest 3 singular values of J are : %.2f %.2f %.2f"%(s[0], s[1], s[2]))
        print("The largest 3 singular values of R are : %.2f %.2f %.2f"%(s2[0], s2[1], s2[2]))
        print("The largest 3 singular values of R_rk1 are : %.2f %.2f %.2f"%(s1[0], s1[1], s1[2]))
        
        

if __name__ == "__main__":
    print("\nTesting QR factorization of J on small matrices.")
    suite_qr = unittest.TestLoader().loadTestsFromTestCase(TestFactorization)
    unittest.TextTestRunner(verbosity=2).run(suite_qr)
    print("\nTesting PCG with direct J.")
    suite_pcg_naive = unittest.TestLoader().loadTestsFromTestCase(TestNaivePCG)
    unittest.TextTestRunner(verbosity=2).run(suite_pcg_naive)
    print("\nTesting triangular solves with R and R^T.")
    suite_tri_solve = unittest.TestLoader().loadTestsFromTestCase(TestTriSolve)
    unittest.TextTestRunner(verbosity=2).run(suite_tri_solve)
    print("\nTesting indirect multiplication methods.")
    suite_mult = unittest.TestLoader().loadTestsFromTestCase(TestMultiply)
    unittest.TextTestRunner(verbosity=2).run(suite_mult)
    print("\nTesting PCG with implicit multiplication")
    suit_pcg_id = unittest.TestLoader().loadTestsFromTestCase(TestPCG_id)
    unittest.TextTestRunner(verbosity=2).run(suit_pcg_id)
    print("\nTesting approximation effectiveness of Qr and thresholding.")
    suite_appox = unittest.TestLoader().loadTestsFromTestCase(TestApproximation)
    unittest.TextTestRunner(verbosity=2).run(suite_appox)
