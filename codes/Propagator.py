import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from Setup import *


class Propagator:
    # This class handles the propagation of the particle through the system.
    # This includes setting the magnetic field and making the particle move through it.

    def __init__(self, magnetic_field: np.array):
        self.magnetic_field = magnetic_field
        self.step_size=None
        self.lowerZBound=None
        self.upperZBound=None
        
    
    # Parameter initialization functions
    def setB0(self,amplitude:float):
        self.magnetic_field = amplitude
        
    def setZBounds(self, lowerZBound: float, upperZBound: float):
        self.lowerZBound = lowerZBound
        self.upperZBound = upperZBound
    
    def setStepSize(self, step_size: float):
        self.step_size = step_size
        
    def computeBField(self, point: np.array):
        _, _, z = point
        constant_region = self.magnetic_field * np.heaviside(z-self.lowerZBound, 1) * np.heaviside(-z+self.upperZBound, 1)
        lower_decay = self.magnetic_field * np.heaviside(-z+self.lowerZBound, 1) * np.exp(-5E-5*(z-self.lowerZBound)**2)
        upper_decay = self.magnetic_field * np.heaviside(z-self.upperZBound, 1) * np.exp(-1E-5*(-z+self.upperZBound)**2)

        # compute the total field in the x direction
        field_x = constant_region + lower_decay + upper_decay
        
        # Assume B= B e_x
        return np.array([field_x, 0.0, 0.0])
    
    # For the propagator we need zeroth and first order derivatives (see Termpaper chapter 3)
    
    def zerothOrderSystemFunction(self, state: np.array, stepSize: float, z:float) -> np.array:
        # this is the evaluation of the system function t at f(x) x=x_now for the zeroth order taylor expansion
        # the state is the five dimensional vector introcuced in the report
        x,y,t_x,t_y,qp = state
        normFactor = np.sqrt(t_x**2 + t_y**2 + 1)
        field = self.computeBField(np.array([x, y, z], dtype=np.float64))
        
        kappa = 2.99792458  # speed of light
        
        # see appendix A of the paper for the equations
        dx_dz= t_x
        dy_dz= t_y
        dtx_dz = kappa * qp * (field[2]*t_y - field[1])*normFactor
        dty_dz = kappa * qp * (field[0] - t_x * field[2])*normFactor
        dqp_dz=0
        
        return np.array([dx_dz, dy_dz, dtx_dz, dty_dz, dqp_dz], dtype=np.float64)
    
    def firstOrderSystemFunction(self, state: np.array, stepSize: float, z:float) -> np.array:
        # this is the jacobian of the system function 
        x, y, t_x, t_y, qp = state
        kappa = 2.99792458  # speed of light
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
        dtxpdqp = kappa * norm_factor * (
            t_y * field[2] - (1 + t_x**2) * field[1] + t_x * t_y * field[0]
        )
        
        dtypdqp = kappa * norm_factor * (
            -t_x * field[2] - (1 + t_y**2) * field[0] - t_x * t_y * field[1]
        )
        
        # Jacobian Matrix
        jacobian = stepSize * np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, dtxpdtx, dtxpdty, dtxpdqp],
            [0, 0, dtypdtx, dtypdty, dtypdqp],
            [0, 0, 0, 0, 0]
        ])
        
        return jacobian
    
    def computeProcessNoise(self, state: np.array, stepSize: float, z: float):
        # compute the process noise for the RK4 Algorithm. Introduce the process noise as matrix dependent on the step
        # size, use detector uncertainties for estimating the values
        
        position_accuracy = 1E-5  # accuracy for position
    
        processCov= stepSize*np.diag([
            position_accuracy,  # x
            position_accuracy,  # y
            position_accuracy,  # tx
            position_accuracy,  # ty
            1E-6   # q/p
        ])
        
        return processCov

    def RK4Propagator(self,state: np.array, covMat: np.array, stepSize: float, z: int):
        identity = np.identity(5)
        
        # compute the different Runge-Kutta constants; use .dot to do the matrix vector multiplication
        # use the implementation for the Jacobian Propagation matrix to compute covariances
        propMat1 = self.firstOrderSystemFunction(state, stepSize, z)
        
        k1 = propMat1.dot(state)
        
        midpoint1 = state + k1 * 0.5
        propMat2_raw = self.firstOrderSystemFunction(midpoint1, stepSize, z + stepSize/2.0)
        k2 = propMat2_raw.dot(midpoint1)
        
        midpoint2 = state + k2 * 0.5
        propMat3_raw = self.firstOrderSystemFunction(midpoint2, stepSize, z + stepSize/2.0)
        k3 = propMat3_raw.dot(midpoint2)
        
        endpoint = state + k3
        propMat4_raw = self.firstOrderSystemFunction(endpoint, stepSize, z + stepSize)
        k4 = propMat4_raw.dot(endpoint)
        
        propMat2 = propMat2_raw @ (identity + 0.5 * propMat1)
        propMat3 = propMat3_raw @ (identity + 0.5 * propMat2)
        propMat4 = propMat4_raw @ (identity + 1.0 * propMat3)
        
        #  Compute the new propagation matrix see RK4 method
        propMat = identity + (
            1/6.0 * propMat1 + 
            1/3.0 * propMat2 + 
            1/3.0 * propMat3 + 
            1/6.0 * propMat4
        )
                
        # RK4 state update
        k1_state = self.zerothOrderSystemFunction(state, stepSize, z)
        k2_state = self.zerothOrderSystemFunction(state + 0.5 * k1_state, stepSize, z + 0.5 * stepSize)
        k3_state = self.zerothOrderSystemFunction(state + 0.5 * k2_state, stepSize, z + 0.5 * stepSize)
        k4_state = self.zerothOrderSystemFunction(state + k3_state, stepSize, z + stepSize)

        new_state = state + (stepSize/6.0) * (k1_state + 2*k2_state + 2*k3_state + k4_state)
        new_covMat = propMat @ covMat @ propMat.T + self.computeProcessNoise(new_state, stepSize, z)
        
        return new_state, new_covMat