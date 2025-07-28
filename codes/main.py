import numpy as np
import matplotlib.pyplot as plt
from Filter import KalmanFilter
from Propagator import Propagator
from Setup import *
import random

momentum = np.array([0.0, 0, 1])*1000  # MeV/c
position = np.array([0.0, 0])

B0 = 1  # Tesla
lowerLimitZ = 500  # cm
upperLimitZ = 1000  # cm
stepSize = 0.01  # cm
charge = -1.0  # in e
initialState, initialCovMat = setParticle(momentum=momentum,startingPosition=position, initialAccuracy=np.array([1e-3, 1e-3, 1e-4, 1e-4, 2e-8]), charge=charge)


zVals = np.linspace(0, 1500, 3001, dtype=np.float64)  # cm
zDet = np.array([250, 300, 350, 400, 1000, 1050, 1100, 1200, 1300])  # cm


fieldPropagator = Propagator(magnetic_field=B0, step_size=stepSize, bounds=(lowerLimitZ, upperLimitZ))
kalmanFilter = KalmanFilter(fieldPropagator)

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

def generate_measurements(realTrack: np.array, zVals: np.array, zDet: np.array, detector_resolution: float = 5e-4) -> np.array:
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

    
def run_tracking_simulation(initialState,detector_resolution):
    realTrack = generate_true_track(fieldPropagator, initialState, initialCovMat, zVals)
    measurements = generate_measurements(realTrack, zVals, zDet, detector_resolution)

    # Setup measurement matrices
    measMat = np.array([[1, 0, 0, 0, 0],    # measure x
                        [0, 1, 0, 0, 0]])   # measure y
    measCovMat = np.array([[detector_resolution**2, 0],
                           [0, detector_resolution**2]])      # 200 μm resolution in x and y

    # Run Kalman filter
    filtStates, filtCov, predStates, predictedCov = kalmanFilter.forwardFilter(measurements, initialCovMat, measMat, measCovMat, initialState, zVals)
    smoothedStates, smoothedCovariances = kalmanFilter.backwardSmoothing(filtStates, filtCov, predStates, predictedCov, zVals, measurements)

    plotTrack(zVals, zDet, realTrack, filtStates, smoothedStates, measurements, savefile="kalman_filter_results.pdf")

    #print("Extrapolating to origin...")
    originState, originCovMat = kalmanFilter.extrapolateToOrigin(smoothedStates, smoothedCovariances, zVals, measurements=measurements)
    print_report(momentum, originState, originCovMat=originCovMat)
    qp_reconstructed = 1/np.abs(originState[4])  # e/MeV
    return qp_reconstructed

def examine_simulation():
    fieldPropagator, kalmanFilter = initialize_world(B0=0.5, lowerLimitZ=500, upperLimitZ=1000, stepSize=1)
    zVals = np.linspace(200, 1400, 3001, dtype=np.float64)  # cm
    zDet = np.array([250, 300, 350, 400, 1000, 1050, 1100, 1200, 1300])  # cm
    
    charges = np.array([-2.0, -1.0, 1.0, 2.0])  # in e
    momenta = np.array([[0, 0.1, 3], [1, 3, 100]])
    colors = ['maroon', 'darkolivegreen', 'navy', 'darkmagenta']
    linestyles = ['-', '--', '-.', ':']
    
    fig, ax = plt.subplots(len(momenta) + 1, figsize=(6,6))  # +1 für das Magnetfeld
    momentum_set=[]
    momentum_inferred=[]
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
    charge_legend1 = [plt.Line2D([0], [0], color=colors[q], lw=2, label=f'Q= {charges[q]}') for q in range(len(charges))]
    charge_legend1.append(plt.Line2D([0], [0], color='black', lw=0, label=f'p={np.linalg.norm(momenta[0]):.2f} GeV/c'))
    ax[0].legend(handles=charge_legend1, loc='upper left', fontsize=8)

    charge_legend2 = [plt.Line2D([0], [0], color=colors[q], lw=2, linestyle=linestyles[q], label=f'Q= {charges[q]}') for q in range(len(charges))]
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
    run_tracking_simulation(initialState=initialState, detector_resolution=5e-2)