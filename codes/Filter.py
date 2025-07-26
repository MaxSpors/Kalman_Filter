import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Propagator import Propagator

class KalmanFilter:
    #  Here we implement the Kalman Filter+ Smoother
    def __init__(self, propagator: Propagator):
        self.propagator = propagator
        self.propagationMatrices = []  # Speichere F-Matrizen

    def makePrediction(self,state: np.array, covMat: np.array, z: np.array,currentZ: float,measuredZ: float):
        # use the theoretical field propagator to make a prediciton of the state at time z    
        currentState = state.copy()
        currentCovMat = covMat.copy()
        
        PropagationMatrices = []
        
        # not every point is measured, so we need to propagate the state until the next measurement
        step_size= z[1]- z[0]
        while currentZ < measuredZ:
            currentState, currentCovMat, PropMatrix = self.propagator.RK4Propagator(currentState, currentCovMat, step_size, currentZ)
            currentZ += step_size
            PropagationMatrices.append(PropMatrix)
        
        total_F = np.identity(5)
        for F in PropagationMatrices:
            total_F = F @ total_F
            
        
        return currentState, currentCovMat, total_F 
    
    
    def forwardFilter(self, measurements: np.array, covMat: np.array, measMat: np.array, measCovMat: np.array, state: np.array, zVals: np.array):
        # Ensure causality by ordering the measurements in z
        time_ordering = np.argsort(measurements[:,2])
        measurements = measurements[time_ordering]

        # Initialize arrays to store results
        n_measurements = len(measurements)
        filteredStates = np.zeros((n_measurements, 5), dtype=np.float64)
        filteredCovariances = np.zeros((n_measurements, 5, 5), dtype=np.float64)
        predictedStates = np.zeros((n_measurements, 5), dtype=np.float64)
        predictedCovariances = np.zeros((n_measurements, 5, 5), dtype=np.float64)
        
        # Store propagation matrices for smoother
        self.propagationMatrices = []

        # Current state and covariance
        currentState = state.copy()
        currentCovMat = covMat.copy()
        currentZ = 0.0

        # Process each measurement
        for i, measurement in enumerate(measurements):
            measuredX, measuredY, measuredZ = measurement
        
            # 1. PREDICT: Propagate state and covariance to the measurement position
            predictedState, predictedCovMat, F = self.makePrediction(currentState, currentCovMat, zVals, currentZ, measuredZ)
            predictedStates[i] = predictedState
            predictedCovariances[i] = predictedCovMat
            
            # Store F matrix for smoother
            self.propagationMatrices.append(F)
        
            # 2. UPDATE: Calculate Kalman gain
            S = measMat @ predictedCovMat @ measMat.T + measCovMat
            KalmanGain = predictedCovMat @ measMat.T @ np.linalg.inv(S)
        
            # Measurement vector [x, y] as that is the only thing we can measure
            measVector = np.array([measuredX, measuredY])
            predictedMeas = measMat @ predictedState
            
            # Innovation (difference between measurement and prediction)
            innovation = measVector - predictedMeas
            updatedState = predictedState + KalmanGain @ innovation
            identity = np.identity(5)
            updatedCovMat = (identity - KalmanGain @ measMat) @ predictedCovMat

            filteredStates[i] = updatedState
            filteredCovariances[i] = updatedCovMat
            
            currentState = updatedState
            currentCovMat = updatedCovMat
            currentZ = measuredZ

        return filteredStates, filteredCovariances, predictedStates, predictedCovariances  
        
    
    def backwardSmoothing(self, filteredStates: np.array, filteredCovariances: np.array, predictedStates: np.array, predictedCovariances: np.array, zVals: np.array, measurements: np.array):
        # initialize the smoothed states and covariances
        n_measurements = len(filteredStates)
        smoothedStates = np.zeros_like(filteredStates)
        smoothedCovariances = np.zeros_like(filteredCovariances)
        
        # Start with the last filtered estimate and work backwards
        smoothedStates[-1] = filteredStates[-1]
        smoothedCovariances[-1] = filteredCovariances[-1]

        for i in range(n_measurements - 2, -1, -1):
            filteredState = filteredStates[i]
            filteredCovariance = filteredCovariances[i]
            
            # Use the stored F matrix from forward pass
            F = self.propagationMatrices[i + 1]  # F from i to i+1
            
            # Compute the smoothing gain (Rauch-Tung-Striebel gain)
            try:
                smoothingGain = filteredCovariance @ F.T @ np.linalg.inv(predictedCovariances[i + 1])
            except np.linalg.LinAlgError:
                smoothingGain = filteredCovariance @ F.T @ np.linalg.pinv(predictedCovariances[i + 1])
                
            smoothedStates[i] = filteredState + smoothingGain @ (smoothedStates[i + 1] - predictedStates[i + 1])
            smoothedCovariances[i] = filteredCovariance + smoothingGain @ (smoothedCovariances[i + 1] - predictedCovariances[i + 1]) @ smoothingGain.T
        
        return smoothedStates, smoothedCovariances
    
    
    def extrapolateToOrigin(self, state: np.array, covMat: np.array, zVals: np.array):
        # From the results of the Kalman filter, extrapolate the track to the origin
        currentState = state.copy()
        currentCovMat = covMat.copy()
        currentZ = zVals[-1]
        
        originalstep = np.abs(zVals[1] - zVals[0])  # original step size
        
        small_step = min(originalstep, abs(currentZ) / 100)
        self.propagator.setStepSize(small_step)  # use a smaller step size for extrapolation

        while abs(currentZ) > 1e-6:
            print(currentZ)
            step_size = -min(originalstep, abs(currentZ))  # ensure we don't overshoot the origin
            # FIX: RK4Propagator gibt 3 Werte zurück!
            currentState, currentCovMat, _ = self.propagator.RK4Propagator(currentState, currentCovMat, step_size, currentZ)
            currentZ += step_size

        originState = currentState
        originCovMat = currentCovMat
        self.propagator.setStepSize(originalstep)  # reset the step size to the original value
        
        # extract parameters at the origin
        originX = originState[0]
        originY = originState[1]
        originZ = currentZ
        origin_qp = originState[4]
        
        p_magnitude = 1.0 / np.abs(origin_qp) if origin_qp != 0 else None
        print('='*50)
        print(f'Extrapolated to origin: x=({originX:.3f}, {originY:.3f}, {originZ:.6f}) cm')
        print(f'Momentum: p={p_magnitude:.3f} GeV/c, q/p={origin_qp:.6f} [e/GeV]')
        
        return originState, originCovMat