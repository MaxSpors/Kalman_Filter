import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple


def setParticle(momentum: np.array, startingPosition: np.array, initialAccuracy:np.array)-> np.array:
    # Set the particle's initial momentum and position
    # also set the initial covariance matrix
    
    momentum = np.array(momentum, dtype=float)*1000 # Convert momentum to MeV/c
    startingPosition = np.array(startingPosition, dtype=float)
    
    # construct the initial covariance matrix
    initial_covMat = np.array([
        [initialAccuracy[0], 0.0, 0.0, 0., 0.],
        [0., initialAccuracy[1], 0., 0., 0.],
        [0., 0., initialAccuracy[2], 0., 0.],
        [0., 0., 0., initialAccuracy[3], 0.],
        [0., 0., 0., 0., initialAccuracy[4]]
    ])
    
    # construct the state vector
    state = np.array([startingPosition[0], startingPosition[1], momentum[0]/momentum[2], momentum[1]/momentum[2], -1./np.linalg.norm(momentum)], dtype=np.float64)

    return state, initial_covMat


def plotTrack(zValues: np.array, DetectorPositions: np.array, realTrack: np.array, states: np.array, measurements: np.array, savefile: str = None) -> None:
    """Plot particle track with measurements and filtered states"""
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot x vs z
    ax[0].plot(zValues[:-1], realTrack[:, 0], label='True Track', color='red', linewidth=2)
    
    # Plot filtered states at measurement positions
    measurement_z = measurements[:, 2]
    ax[0].plot(measurement_z, states[:, 0], 'o-', label='Filtered Track', color='blue', markersize=6)
    
    # Plot measurements
    ax[0].plot(measurements[:, 2], measurements[:, 0], 'x', label='Measurements', color='green', markersize=8, markeredgewidth=2)
    
    # Plot detector positions
    for det_z in DetectorPositions:
        ax[0].axvline(x=det_z, color='gray', linestyle='--', alpha=0.5)
    
    ax[0].set_xlabel('z [cm]')
    ax[0].set_ylabel('x [cm]')
    ax[0].set_title('Particle Track - X Coordinate')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Plot y vs z
    ax[1].plot(zValues[:-1], realTrack[:, 1], label='True Track', color='red', linewidth=2)
    
    # Plot filtered states at measurement positions
    ax[1].plot(measurement_z, states[:, 1], 'o-', label='Filtered Track', color='blue', markersize=6)
    
    # Plot measurements
    ax[1].plot(measurements[:, 2], measurements[:, 1], 'x', label='Measurements', color='green', markersize=8, markeredgewidth=2)
    
    # Plot detector positions
    for det_z in DetectorPositions:
        ax[1].axvline(x=det_z, color='gray', linestyle='--', alpha=0.5)
    
    ax[1].set_xlabel('z [cm]')
    ax[1].set_ylabel('y [cm]')
    ax[1].set_title('Particle Track - Y Coordinate')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {savefile}")
    
    plt.show()