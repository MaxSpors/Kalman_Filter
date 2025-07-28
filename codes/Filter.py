import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Propagator import Propagator

class KalmanFilter:
    def __init__(self, propagator: Propagator):
        self.propagator = propagator
        self.propagationMatrices = [] # save computational effort by storing the propagation matrices

    def makePrediction(self,state, covMat, z,currentZ,measuredZ):
        currentState = state.copy()
        currentCovMat = covMat.copy()
        PropagationMatrices = []       
        
        step_size= z[1]- z[0]
        while currentZ < measuredZ-1e-2:
            currentState, currentCovMat, PropMatrix = self.propagator.RK4Propagator(currentState, currentCovMat, step_size, currentZ)
            currentZ += step_size
            PropagationMatrices.append(PropMatrix)
        
        total_F = np.identity(5)
        for F in PropagationMatrices:
            total_F = total_F @ F
        return currentState, currentCovMat, total_F 
    
    def forwardFilter(self, measurements, covMat, measMat, measCovMat, state, zVals):
        time_ordering = np.argsort(measurements[:,2])
        measurements = measurements[time_ordering]

        n_measurements = len(measurements)
        filteredStates = np.zeros((n_measurements, 5), dtype=np.float64)
        filteredCovariances = np.zeros((n_measurements, 5, 5), dtype=np.float64)
        predictedStates = np.zeros((n_measurements, 5), dtype=np.float64)
        predictedCovariances = np.zeros((n_measurements, 5, 5), dtype=np.float64)
        
        self.propagationMatrices = []

        currentState = state.copy()
        currentCovMat = covMat.copy()
        currentZ = 0.0

        # Process each measurement
        for i, measurement in enumerate(measurements):
            measuredX, measuredY, measuredZ = measurement
        
            # 1. PREDICT: Propagate state and covariance
            predictedState, predictedCovMat, F = self.makePrediction(currentState, currentCovMat, zVals, currentZ, measuredZ)
            predictedStates[i] = predictedState
            predictedCovariances[i] = predictedCovMat
            
            # Store F matrix for smoother
            self.propagationMatrices.append(F)
        
            # 2. UPDATE: Calculate Kalman gain
            S = measMat @ predictedCovMat @ measMat.T + measCovMat
            KalmanGain = predictedCovMat @ measMat.T @ np.linalg.inv(S)

            measVector = np.array([measuredX, measuredY])
            predictedMeas = measMat @ predictedState
            
            # Innovation (difference between measurement and prediction)
            innovation = measVector - predictedMeas
            updatedState = predictedState + KalmanGain @ innovation
            identity = np.identity(5)
            # Einfachere, numerisch stabilere Form:
            updatedCovMat = (identity - KalmanGain @ measMat) @ predictedCovMat
            filteredStates[i] = updatedState
            filteredCovariances[i] = updatedCovMat
            
            currentState = updatedState
            currentCovMat = updatedCovMat
            currentZ = measuredZ

        return filteredStates, filteredCovariances, predictedStates, predictedCovariances  
            
    def backwardSmoothing(self, filteredStates: np.array, filteredCovariances: np.array, predictedStates: np.array, predictedCovariances: np.array, zVals: np.array, measurements: np.array):
        n_measurements = len(filteredStates)
        smoothedStates = np.zeros_like(filteredStates, dtype=np.float64)
        smoothedCovariances = np.zeros_like(filteredCovariances, dtype=np.float64)

        # Start with the last filtered estimate and work backwards
        smoothedStates[-1] = filteredStates[-1]
        smoothedCovariances[-1] = filteredCovariances[-1]

        for i in range(n_measurements - 2, -1, -1):
            filteredState = filteredStates[i]
            filteredCovariance = filteredCovariances[i]
            
            F = self.propagationMatrices[i+1]
            smoothingGain = filteredCovariance @ F.T @ np.linalg.inv(predictedCovariances[i + 1])
            smoothedStates[i] = filteredState + smoothingGain @ (smoothedStates[i + 1] - predictedStates[i + 1])
            smoothedCovariances[i] = filteredCovariance + smoothingGain @ (smoothedCovariances[i + 1] - predictedCovariances[i + 1]) @ smoothingGain.T
            smoothedCovariances[i] = (smoothedCovariances[i] + smoothedCovariances[i].T) / 2

        return smoothedStates, smoothedCovariances

    def extrapolateToOrigin(self, smoothedStates: np.array, smoothedCovariances: np.array, zVals: np.array, measurements: np.array):
        # Find the measurement closest to the origin, for the smoothed data this is the 'best' estimate for back propagation
        measurement_z_positions = measurements[:, 2]
        closest_to_origin_idx = np.argmin(np.abs(measurement_z_positions))
        
        best_state = smoothedStates[closest_to_origin_idx].copy()
        best_cov = smoothedCovariances[closest_to_origin_idx].copy()
        start_z = measurement_z_positions[closest_to_origin_idx]
        
        # Initialize extrapolation
        currentState = best_state.copy()
        currentCovMat = best_cov.copy()
        currentZ = start_z
        
        step_size = -np.abs(zVals[1] - zVals[0])
        # Extrapolation loop        
        while currentZ > 1e-6:
            currentState, currentCovMat, _ = self.propagator.RK4Propagator(currentState, currentCovMat, step_size, currentZ, includeProcessNoise=False)
            currentZ += step_size
        return currentState, currentCovMat