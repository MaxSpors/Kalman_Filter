import numpy as np
import matplotlib.pyplot as plt

def setParticle(momentum, startingPosition, initialAccuracy, charge=-1.0):
    momentum = np.array(momentum, dtype=np.float64)
    startingPosition = np.array(startingPosition, dtype=np.float64)

    initial_covMat = np.diag([initialAccuracy[0], initialAccuracy[1], initialAccuracy[2], initialAccuracy[3], initialAccuracy[4]])
    state = np.array([startingPosition[0], startingPosition[1], momentum[0]/momentum[2], momentum[1]/momentum[2], charge/np.linalg.norm(momentum)], dtype=np.float64)

    return state, initial_covMat

def plotTrack(zValues, DetPos, realTrack, states, smoothed_states, measurements, savefile=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(zValues[:-1], realTrack[:, 1], label='True', color='red', linewidth=2)
    ax.plot(measurements[:, 2], states[:, 1], '--', label='Filtered', color='blue', linewidth=2)
    ax.plot(measurements[:, 2], smoothed_states[:, 1], ':', label='Smoothed', color='orange', linewidth=2)
    ax.errorbar(measurements[:, 2], measurements[:, 1], yerr=70e-4, fmt='.', label='Measured', color='green', markersize=8, markeredgewidth=2)
    # Detector positions
    for det_z in DetPos:
        ax.axvline(x=det_z, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('z [cm]', fontsize=12)
    ax.set_ylabel('y [cm]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)    
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')    
    plt.show()


def print_report(initialMomentum, originState, originCovMat):
    print('='*60)
    print('RECONSTRUCTION REPORT')
    print('='*60)
    print(f'  |p|_0= { np.linalg.norm(initialMomentum):.3f} MeV/c')
    
    qp_reconstructed = 1/np.abs(originState[4])  # e/MeV
    sigma_qp = np.sqrt(originCovMat[4, 4]) * qp_reconstructed**2  # e/MeV

    print(f'  |p|_est = {qp_reconstructed:.3f} ± {sigma_qp:.3f} MeV/c')




def plotHistogram(data, xlabel, ylabel, savefile=None, mean=None, std=None):
    plt.figure(figsize=(6, 6))
    plt.hist(data, color='blue', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if mean is not None and std is not None:
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
        plt.axvline(mean + std, color='orange', linestyle='dashed', linewidth=1)
        plt.axvline(mean - std, color='orange', linestyle='dashed', linewidth=1)
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()