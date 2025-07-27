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

def init_state(momentum: np.array, position: np.array, charge: float) -> Tuple[np.array, np.array]:
    """Initialize particle state and covariance matrix"""
    state, covMat = setParticle(
        momentum=momentum,  # in GeV/c
        startingPosition=position,  # in cm
        initialAccuracy=np.array([1e-1, 1e-1, 1e-1, 1e-1, 0.05*np.linalg.norm(momentum)/1000]),  # in cm and 1/MeV
        charge=charge
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

def generate_measurements(realTrack: np.array, zVals: np.array, zDet: np.array, detector_resolution: float = 70e-4) -> np.array:
    measurements = np.empty([0, 3])

    for i, z in enumerate(zVals[:-1]):
        # Check if this z position corresponds to a detector
        detector_indices = np.where(np.abs(z - zDet) < 0.5)[0]
        if len(detector_indices) > 0:
            # Get true state at this position
            true_state = realTrack[i]
            
            # Add detector resolution (Gaussian noise)
            dx = random.gauss(0, detector_resolution)
            dy = random.gauss(0, detector_resolution)
            
            measurement = np.array([[true_state[0] + dx, true_state[1] + dy, z]])
            measurements = np.concatenate((measurements, measurement), axis=0)
    
    return measurements

def calculate_curvature_radius(realTrack: np.array, zVals: np.array, interval: Tuple[float, float]) -> float:
    # Test the implementation for checking with p= 0.3 B R
    z_min, z_max = interval
    indices = np.where((zVals >= z_min) & (zVals <= z_max))[0]
        
    x = realTrack[indices, 0]
    y = realTrack[indices, 1]
    z = zVals[indices]
    
    # numerical derivatives
    dx = np.gradient(x, z)
    dy = np.gradient(y, z)
    ddx = np.gradient(dx, z)
    ddy = np.gradient(dy, z)
    
    # compute curvature
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # compute radius
    radius = 1 / np.mean(curvature)
    
    return radius



def print_report(initialMomentum: np.array, originState: np.array, originCovMat: np.array):
    print('='*60)
    print('TRACKING RECONSTRUCTION REPORT')
    print('='*60)
    
    # Initial values
    initial_p_magnitude = np.linalg.norm(initialMomentum)
    print(f'\nInitial Parameters:')
    print(f'  Momentum magnitude: {initial_p_magnitude:.3f} GeV/c')
    print(f'  Direction: p={initialMomentum} GeV/c')
    
    # Reconstructed values
    qp_reconstructed = originState[4]  # e/MeV
    p_reconstructed_GeV = 1.0 / abs(qp_reconstructed) / 1000.0 if qp_reconstructed != 0 else None
    sigma_qp = np.sqrt(originCovMat[4, 4])  # e/MeV
    sigma_p_GeV = sigma_qp / (qp_reconstructed**2) / 1000.0 if qp_reconstructed != 0 else None
    
    print(f'\nReconstructed Parameters:')
    print(f'  p= ({p_reconstructed_GeV:.3f} +- {sigma_p_GeV:.3f}) GeV/c')
    print('='*60)

def run_tracking_simulation():

    fieldPropagator, kalmanFilter = initialize_world(B0=0.5, lowerLimitZ=500, upperLimitZ=1000, stepSize=1)

    # Initialize particle
    momentum = np.array([0.0, 0.1, 1])  # GeV/c in x, y, z
    position = np.array([0.0, 0.0])       # cm, starting position

    initialState, initialCovMat = init_state(momentum, position,charge=-1.0)  # Charge in e

    # Setup tracking parameters
    zVals = np.linspace(0, 1500, 3001, dtype=np.float64)  # cm
    zDet = np.array([250, 300, 350, 400, 1000, 1050, 1100, 1200, 1300])  # cm
    
    realTrack = generate_true_track(fieldPropagator, initialState, initialCovMat, zVals)
    measurements = generate_measurements(realTrack, zVals, zDet, detector_resolution=0.001)  # 100 μm resolution
    
    # Setup measurement matrices
    measMat = np.array([[1, 0, 0, 0, 0],    # measure x
                        [0, 1, 0, 0, 0]])   # measure y
    measCovMat = np.array([[0.01, 0],
                           [0, 0.01]])      # 100 μm resolution in x and y
    
    # Run Kalman filter
    filteredStates, filteredCovariances, predictedStates, predictedCovariances = kalmanFilter.forwardFilter(measurements, initialCovMat, measMat, measCovMat, initialState, zVals)
    smoothedStates, smoothedCovariances = kalmanFilter.backwardSmoothing(filteredStates, filteredCovariances, predictedStates, predictedCovariances, zVals, measurements)
    
    plotTrack(zVals, zDet, realTrack, filteredStates, measurements, savefile="kalman_filter_results.png")
    plotTrack(zVals, zDet, realTrack, smoothedStates, measurements, savefile="kalman_smoothed_results.png")
    
    print("Extrapolating to origin...")
    originState, originCovMat = kalmanFilter.extrapolateToOrigin(smoothedStates, smoothedCovariances, zVals, measurements=measurements)

    print_report(momentum, originState, originCovMat=originCovMat)


def examine_simulation():
    fieldPropagator, kalmanFilter = initialize_world(B0=0.5, lowerLimitZ=500, upperLimitZ=1000, stepSize=1)
    zVals = np.linspace(100, 1500, 3001, dtype=np.float64)  # cm
    zDet = np.array([250, 300, 350, 400, 1000, 1050, 1100, 1200, 1300])  # cm
    
    charges = np.array([-2.0, -1.0, 1.0, 2.0])  # in e
    momenta = np.array([[0.0, 0.5, 2], [0.5, 0.1, 10]])
    colors = ['maroon', 'darkolivegreen', 'navy', 'darkmagenta']
    linestyles = ['-', '--', '-.', ':']
    
    fig, ax = plt.subplots(len(momenta) + 1, figsize=(9, 8))  # +1 für das Magnetfeld
    
    for p, momentum in enumerate(momenta):
        for q, charge in enumerate(charges):
            initialState, initialCovMat = init_state(momentum, np.array([0.0, 0.0]), charge)
            realTrack = generate_true_track(fieldPropagator, initialState, initialCovMat, zVals)
            measurements = generate_measurements(realTrack, zVals, zDet, detector_resolution=0.001)
            # Plot all charges for the same momentum in the same subplot
            ax[p].plot(zVals[:-1], realTrack[:, 1], color=colors[q], linestyle=linestyles[q], linewidth=1)
            ax[p].errorbar(measurements[:, 2], measurements[:, 1], yerr=70e-4, fmt='o', markersize=3, color=colors[q])
            
        ax[p].set_xlabel('z [cm]')
        ax[p].set_ylabel('y [cm]')
        ax[p].grid(True, alpha=0.3)
    
    # Manuelle Legende für Ladungen
    charge_legend1 = [plt.Line2D([0], [0], color=colors[q], lw=2, label=f'Charge: {charges[q]}') for q in range(len(charges))]
    charge_legend1.append(plt.Line2D([0], [0], color='black', lw=0, label=f'p={np.linalg.norm(momenta[0]):.2f} GeV/c'))
    ax[0].legend(handles=charge_legend1, loc='upper left', fontsize=8)

    charge_legend2 = [plt.Line2D([0], [0], color=colors[q], lw=2, linestyle=linestyles[q], label=f'Charge: {charges[q]}') for q in range(len(charges))]
    charge_legend2.append(plt.Line2D([0], [0], color='black', lw=0, label=f'p={np.linalg.norm(momenta[1]):.2f} GeV/c'))
    ax[1].legend(handles=charge_legend2, loc='upper left', fontsize=8)

    # Magnetfeld als Funktion von z
    BField = np.array([fieldPropagator.computeBField([0, 0, z])[0] for z in zVals])  # Nur x-Komponente
    ax[-1].plot(zVals, BField, label='Magnetic Field (B_x)', color='navy', linewidth=2)
    ax[-1].set_xlabel('z [cm]')
    ax[-1].set_ylabel('B [T]')
    ax[-1].legend()
    ax[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("examine_simulation_results.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    

if __name__ == "__main__":
    run_tracking_simulation()
    examine_simulation()