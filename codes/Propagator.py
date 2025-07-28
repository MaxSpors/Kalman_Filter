import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from Setup import *


class Propagator:
    def __init__(self, magnetic_field, step_size, bounds):
        self.magnetic_field = magnetic_field
        self.step_size = step_size
        self.lowerZBound = bounds[0]
        self.upperZBound = bounds[1]

    # Process Noise estimation per step    
    def HighlandEstimate(self, momentum, step_size, mass, rad_length= 1.176e04):
        beta = momentum / np.sqrt(momentum**2 + mass**2)
        theta_rms = 13.6 / (momentum * beta) * np.sqrt(step_size / rad_length)
        return theta_rms
    
    def computeProcessNoise(self, state, stepSize):
        momentum=np.abs(1.0/state[4])
        theta_rms=self.HighlandEstimate(momentum=momentum, step_size=stepSize, mass=0.511)
        position_accuracy = theta_rms *(stepSize)
        processCov= np.abs(stepSize)*np.diag([position_accuracy, position_accuracy, theta_rms, theta_rms, 1e-10])
        return processCov

    def computeBField(self, point):
        _, _, z = point
        constant_region = self.magnetic_field * np.heaviside(z-self.lowerZBound, self.magnetic_field) * np.heaviside(-z+self.upperZBound, self.magnetic_field)
        lower_decay = self.magnetic_field * np.heaviside(-z+self.lowerZBound, self.magnetic_field) * np.exp(-5E-5*(z-self.lowerZBound)**2)
        upper_decay = self.magnetic_field * np.heaviside(z-self.upperZBound, self.magnetic_field) * np.exp(-1E-5*(-z+self.upperZBound)**2)
        
        # Assume B= B e_x
        field_x = constant_region + lower_decay + upper_decay
        return np.array([field_x, 0.0, 0.0])
    
    # For the propagator we need zeroth and first order derivatives (see Termpaper chapter 3)
    
    def stateProp(self, state, stepSize, z):
        x,y,t_x,t_y,qp = state
        normFactor = np.sqrt(t_x**2 + t_y**2 + 1)
        field = self.computeBField(np.array([x, y, z], dtype=np.float64))
        
        kappa = 2.99792458  # speed of light

        # This is a direct consequence of the Lorentz force Ansatz
        dtx_dz = kappa * qp * (field[2]*t_y - field[1])*normFactor
        dty_dz = kappa * qp * (field[0] - t_x * field[2])*normFactor
        return np.array([t_x, t_y, dtx_dz, dty_dz, 0], dtype=np.float64)

    def Jacobian(self, state, stepSize, z):
        x, y, t_x, t_y, qp = state
        kappa = 2.99792458
        field = self.computeBField(np.array([x, y, z], dtype=np.float64))        
        norm_factor = np.sqrt(t_x**2 + t_y**2 + 1)

        # dt_x/dx derivative
        dtxpdtx = kappa * qp * (
            field[0] * t_y * (2*t_x**2 + t_y**2 + 1) - 
            field[1] * t_x * (3*t_x**2 + 2*t_y**2 + 3) + 
            field[2] * t_x * t_y
        ) / norm_factor
        
        # dt_x/dty 
        dtxpdty = kappa * qp * (
            field[0] * (t_x**3 + 2*t_x*t_y**2 + t_x) - 
            field[1] * (t_x**2 + 1) * t_y + 
            field[2] * (t_x**2 + 2*t_y**2 + 1)
        ) / norm_factor
        
        # dt_y/dtx
        dtypdtx = kappa * qp * (
            -field[0] * t_x * (t_y**2 + 1) - 
            field[1] * t_y * (2*t_x**2 + t_y**2 + 1) - 
            field[2] * (2*t_x**2 + t_y**2 + 1)
        ) / norm_factor
        
        # dt_y/dty
        dtypdty = -kappa * qp * (
            field[0] * (2*t_x**2*t_y + 3*t_y**3 + 3*t_y) + 
            field[1] * (t_x**3 + 2*t_x*t_y**2 + t_x) + 
            field[2] * t_x * t_y
        ) / norm_factor
        
        # q/p
        dtxpdqp = kappa * norm_factor * (t_y * field[2] - (1 + t_x**2) * field[1] + t_x * t_y * field[0])
        
        dtypdqp = kappa * norm_factor * (-t_x * field[2] - (1 + t_y**2) * field[0] - t_x * t_y * field[1])
        
        jacobian = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, dtxpdtx, dtxpdty, dtxpdqp],
            [0, 0, dtypdtx, dtypdty, dtypdqp],
            [0, 0, 0, 0, 0]
        ])
        
        return jacobian
    
    def RK4Propagator(self,state, covMat, stepSize, z, includeProcessNoise = True):
        # state update
        k1_state = self.stateProp(state, stepSize, z)
        k2_state = self.stateProp(state + 0.5 * k1_state, stepSize, z + 0.5 * stepSize)
        k3_state = self.stateProp(state + 0.5 * k2_state, stepSize, z + 0.5 * stepSize)
        k4_state = self.stateProp(state + k3_state, stepSize, z + stepSize)
        new_state = state + (stepSize/6.0) * (k1_state + 2*k2_state + 2*k3_state + k4_state)
        
        # Covariance Update
        F = np.identity(5)
        
        # k- States for F- Matrix
        J1 = self.Jacobian(state, stepSize, z)
        k1_F = J1 @ F
        
        state2 = state + 0.5 * k1_state
        state3 = state + 0.5 * k2_state
        state4 = state + k3_state

        J2 = self.Jacobian(state2, stepSize, z + 0.5 * stepSize)
        k2_F = J2 @ (F + 0.5 * stepSize * k1_F)
        
        J3 = self.Jacobian(state3, stepSize, z + 0.5 * stepSize)
        k3_F = J3 @ (F + 0.5 * stepSize * k2_F)
        
        J4 = self.Jacobian(state4, stepSize, z + stepSize)
        k4_F = J4 @ (F + stepSize * k3_F)
        
        propMat = F + (stepSize/6.0) * (k1_F + 2*k2_F + 2*k3_F + k4_F)
        
        # Covariance Update: Smoothing does not include process noise
        if includeProcessNoise:
            new_covMat = propMat @ covMat @ propMat.T + self.computeProcessNoise(new_state, stepSize)
        else:
            new_covMat = propMat @ covMat @ propMat.T
        
        return new_state, new_covMat, F