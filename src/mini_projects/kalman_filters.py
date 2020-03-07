# Write a function 'kalman_filter' that implements a multi-
# dimensional Kalman Filter for the example given

from math import *


class matrix:
    
    # implements basic operations of a matrix class
    
    def __init__(self, value):
        self.value = value
        self.dimx = len(value)
        self.dimy = len(value[0])
        if value == [[]]:
            self.dimx = 0
        self.shape = [self.dimx, self.dimy]
        # self.init_shape(value)
        
    def init_shape(self, value):
        dimx = len(value)
        dimy = len(value[0])
        self.shape = [dimx, dimy]
        
    def zero(self, dimx, dimy):
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError ("Invalid size of matrix")
        else:
            self.dimx = dimx
            self.dimy = dimy
            self.value = [[0 for row in range(dimy)] for col in range(dimx)]
    
    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError ("Invalid size of matrix")
        else:
            self.dimx = dim
            self.dimy = dim
            self.value = [[0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1
    
    def show(self):
        for i in range(self.dimx):
            print(self.value[i])
        print(' ')
    
    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError ("Matrices must be of equal dimensions to add")
        else:
            # add if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] + other.value[i][j]
            res.init_shape(res.value)
            return res
    
    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError ("Matrices must be of equal dimensions to subtract")
        else:
            # subtract if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] - other.value[i][j]

            res.init_shape(res.value)
            return res
    
    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:
            raise ValueError ("Matrices must be m*n and n*p to multiply")
        else:
            # multiply if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, other.dimy)
            for i in range(self.dimx):
                for j in range(other.dimy):
                    for k in range(self.dimy):
                        res.value[i][j] += self.value[i][k] * other.value[k][j]
            res.init_shape(res.value)
            return res
    
    def transpose(self):
        # compute transpose
        res = matrix([[]])
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        res.init_shape(res.value)
        return res
    
    # Thanks to Ernesto P. Adorio for use of Cholesky and CholeskyInverse functions
    
    def Cholesky(self, ztol=1.0e-5):
        # Computes the upper triangular Cholesky factorization of
        # a positive definite matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        for i in range(self.dimx):
            S = sum([(res.value[k][i]) ** 2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else:
                if d < 0.0:
                    raise ValueError ("Matrix not positive-definite")
                res.value[i][i] = sqrt(d)
            for j in range(i + 1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(self.dimx)])
                if abs(S) < ztol:
                    S = 0.0
                try:
                    res.value[i][j] = (self.value[i][j] - S) / res.value[i][i]
                except:
                    raise ValueError ("Zero diagonal")

        res.init_shape(res.value)
        return res
    
    def CholeskyInverse(self):
        # Computes inverse of matrix given its Cholesky upper Triangular
        # decomposition of matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k] * res.value[j][k] for k in range(j + 1, self.dimx)])
            res.value[j][j] = 1.0 / tjj ** 2 - S / tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = -sum(
                        [self.value[i][k] * res.value[k][j] for k in range(i + 1, self.dimx)]) / self.value[i][i]
        res.init_shape(res.value)
        return res
    
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res
    
    def __repr__(self):
        return repr(self.value)


########################################

# Implement the filter function below

def kalman_filter(x, P):
    for n in range(len(measurements)):
        print("Running Time Step: ", n)
        z = matrix([[measurements[n]]])
        print("Shape of measurement: ", z.shape)
        
        # ----------------------------------------------------------------------
        # Measurement Update
        # ----------------------------------------------------------------------
        # Step 1: Determine Error: measurement - prediction
        y = z - H.__mul__(x)
        print("[Shape] y (error-in-estimate) = ", y.shape)
        
        # Step 2: Calculate the System Error (Prediction - Measurement)
        S = (H.__mul__(P).__mul__(H.transpose())).__add__(R)
        print("[Shape] S (error-in-prediction) = ", P.shape)
        
        # Step 3: Calculate Kalman Gain
        K = (P.__mul__(H.transpose())).__mul__(S.inverse())
        print("[Shape] K (kalman-gain) = ", K.shape)
        
        # Step 4: Calculate Prediction Update
        x = x.__add__(K.__mul__(y))
        P = (I.__sub__(K.__mul__(H))).__mul__(P)

        # ----------------------------------------------------------------------
        # Predictions (These becomes prior for the next time steps)
        # ----------------------------------------------------------------------
        x = (F.__mul__(x)).__add__(u)
        P = F.__mul__(P).__mul__(F.transpose())
        print("[Shape] x (prediction-estimate) = ", x.shape)
        print("[Shape] P (prediction-covariance) = ", P.shape)

    return x, P


############################################
### use the code below to test your filter!
############################################
"""
Kalman Filters in 1D: (Only deals with Location)
    We basically need three things with measurement:
    1. Priors (Our prior beliefs about the measurement and its uncertainity)
    2. Measurements: likelihood: our current measurements and their uncertainities
    3. Motion: For our case the object is not stable, so while calculating the predictions we have to take into account, the
       motion and its uncertainity.
       
    At each timestep, we compute:
    1. Prediction (posterior)
        In normal bayes law, posterior = prior*likelihood
    2. Wrap up kalman filters in 1d
        At timestep t:
            prediction(t) = prior(t)*measurement(t) + motion(t)
                 -> where prior(t) = prediction(t-1)
                 
kalman Filters in 2D: (Deals with Location and velocity -> where velocity is unknown)




"""
# We define the Priors N(x, P)
x = matrix([[0.], [0.]])  # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]])  # initial uncertainty

# We define the Measurement N(m, 1)
measurements = z = [1, 2, 3]

# We define the External Motion ()
u = matrix([[0.], [0.]])  # external motion

# We define the transition functions: These are constants and do not change
# These functions generalize the 1D arithmatic operations for 2D
F = matrix([[1., 1.], [0, 1.]])  # next state transition function
H = matrix([[1., 0.]])  # 1-> because we observe measurement, 0-> because we don't observe velocity
R = matrix([[1.]])  # measurement uncertainty:
I = matrix([[1., 0.], [0., 1.]])  # identity matrix


print("[Shape] x (prior-estimate) = ", x.shape)
print("[Shape] P (prior-covariance) = ", P.shape)
print("[Shape] u (external-motion) = ", u.shape)
print("[Shape] F (state-transition-matrix) = ", F.shape)
print("[Shape] H (measurement-function) = ", H.shape)
print("[Shape] R (measurement-noise) = ", R.shape)
print("[Shape] I (identity-matrix) = ", I.shape)

print(kalman_filter(x, P))
# output should be:
# x: [[3.9996664447958645], [0.9999998335552873]]
# P: [[2.3318904241194827, 0.9991676099921091], [0.9991676099921067, 0.49950058263974184]]
