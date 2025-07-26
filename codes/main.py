import numpy as np
import matplotlib.pyplot as plt
from Filter import KalmanFilter
from Propagator import Propagator
from Setup import *
import random


def initialize_world(B0: float, lowerLimitZ: float, upperLimitZ: float, stepSize: float) -> Tuple[Propagator, KalmanFilter]:
    """Initialize the physics world with magnetic field and propagator"""
    fieldPropagator = Propagator(magnetic_field=B0)
    fieldPropagator.setB0(B0)  # in Tesla
    fieldPropagator.setZBounds(lowerLimitZ, upperLimitZ)  # in cm
    fieldPropagator.setStepSize(stepSize)  # in cm
    
    kalmanFilter = KalmanFilter(fieldPropagator)
    
    return fieldPropagator, kalmanFilter


def init_state(momentum: np.array, position: np.array) -> Tuple[np.array, np.array]:
    """Initialize particle state and covariance matrix"""
    state, covMat = setParticle(
        momentum=momentum,  # in GeV/c
        startingPosition=position,  # in cm
        initialAccuracy=np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-3])  # in cm and 1/GeV
    )
    return state, covMat


def generate_true_track(fieldPropagator: Propagator, initialState: np.array, initialCovMat: np.array, zVals: np.array) -> np.array:
    state = initialState.copy()
    covMat = initialCovMat.copy()
    realTrack = np.empty([0, 5])
    
    for i in range(len(zVals) - 1):
        z = zVals[i]
        stepSize = zVals[i + 1] - zVals[i]
        
        # Propagate state (ignore covariance and F matrix for true track)
        state, _, _ = fieldPropagator.RK4Propagator(state, covMat, stepSize, z)
        realTrack = np.concatenate((realTrack, np.reshape(state, (1, 5))), axis=0)
    
    return realTrack


def generate_measurements(realTrack: np.array, zVals: np.array, zDet: np.array, detector_resolution: float = 1e-4) -> np.array:
    measurements = np.empty([0, 3])

    for i, z in enumerate(zVals[:-1]):
        # Check if this z position corresponds to a detector
        detector_indices = np.where(np.abs(z - zDet) < 1e-6)[0]
        
        if len(detector_indices) > 0:
            # Get true state at this position
            true_state = realTrack[i]
            
            # Add detector resolution (Gaussian noise)
            dx = random.gauss(0, detector_resolution)
            dy = random.gauss(0, detector_resolution)
            
            measurement = np.array([[true_state[0] + dx, true_state[1] + dy, z]])
            measurements = np.concatenate((measurements, measurement), axis=0)
    
    return measurements


def run_tracking_simulation():
    """Main simulation function"""
    
    # Initialize world
    fieldPropagator, kalmanFilter = initialize_world(
        B0=0.5,           # Tesla
        lowerLimitZ=500,    # cm
        upperLimitZ=1000, # cm
        stepSize=1      # cm
    )
    
    # Initialize particle
    momentum = np.array([0.0, 0.01, 1])  # GeV/c in x, y, z
    position = np.array([0.0, 0.0])       # cm, starting position

    initialState, initialCovMat = init_state(momentum, position)

    # Setup tracking parameters
    zVals = np.linspace(0, 1500, 3001, dtype=np.float64)  # cm
    zDet = np.array([250, 300, 350, 400, 1000, 1050, 1100, 1200, 1300])  # cm
    
    # Generate true track
    print("Generating true particle track...")
    realTrack = generate_true_track(fieldPropagator, initialState, initialCovMat, zVals)
    
    # Generate measurements
    print("Generating detector measurements...")
    measurements = generate_measurements(realTrack, zVals, zDet, detector_resolution=0.001)  # 100 μm resolution
    
    # Setup measurement matrices
    measMat = np.array([[1, 0, 0, 0, 0],    # measure x
                        [0, 1, 0, 0, 0]])   # measure y
    measCovMat = np.array([[1e-4, 0],       # measurement covariance (100 μm)²
                           [0, 1e-4]])
    
    # Run Kalman filter
    print("Running forward Kalman filter...")
    filteredStates, filteredCovariances, predictedStates, predictedCovariances = kalmanFilter.forwardFilter(
        measurements, initialCovMat, measMat, measCovMat, initialState, zVals
    )
    
    # Run smoother
    print("Running backward smoother...")
    smoothedStates, smoothedCovariances = kalmanFilter.backwardSmoothing(
        filteredStates, filteredCovariances, predictedStates, predictedCovariances, zVals, measurements
    )
    
    # Plot results
    print("Plotting results...")
    plotTrack(zVals, zDet, realTrack, filteredStates, measurements, savefile="kalman_filter_results.png")
    plotTrack(zVals, zDet, realTrack, smoothedStates, measurements, savefile="kalman_smoothed_results.png")
    
    # Extrapolate to origin
    print("Extrapolating to origin...")
    originState, originCovMat = kalmanFilter.extrapolateToOrigin(smoothedStates, smoothedCovariances, zVals, measurements=measurements)

    # Print results
    print(f"\nResults:")
    print(f"Initial momentum: {momentum} GeV/c")
    print(f"Reconstructed q/p at origin: {originState[4]:.6f} e/GeV")
    print(f"Reconstructed momentum: {1e-3/abs(originState[4]):.3f} GeV/c")
    print(f"Position at origin: ({originState[0]:.3f}, {originState[1]:.3f}) cm")


if __name__ == "__main__":
    run_tracking_simulation()