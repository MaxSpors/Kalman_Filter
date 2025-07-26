import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple


def setParticle(momentum: np.array, startingPosition: np.array, initialAccuracy:np.array)-> np.array:
    # Set the particle's initial momentum and position
    # also set the initial covariance matrix
    
    momentum = np.array(momentum, dtype=float)
    startingPosition = np.array(startingPosition, dtype=float)
    
    # construct the initial covariance matrix
    initial_covMat = np.array([
        [initialAccuracy[0], 0.0, 0.0, 0., 0.],
        [0., initialAccuracy[1], 0., 0., 0.],
        [0., 0., initialAccuracy[2], 0., 0.],
        [0., 0., 0., initialAccuracy[3], 0.],
        [0., 0., 0., 0., 1/2*np.linalg.norm(momentum)]
    ])
    
    # construct the state vector
    state = np.array([startingPosition[0], startingPosition[1], momentum[0]/momentum[2], momentum[1]/momentum[2], -1./np.linalg.norm(momentum)], dtype=np.float64)

    return state, initial_covMat


def plotTrack(zValues: np.array, DetectorPositions: np.array, realTrack: np.array, states:np.array, measurements: np.array, savefile: str= None) -> None:
    # Plot the track of the particle one time the x-z track and one time the y-z track
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # x-z Track
    ax[0].set_title("x-z Track")
    ax[0].set_xlabel("z [cm]")
    ax[0].set_ylabel("x [cm]")
    ax[0].plot(zValues, states[:, 0], label='Filtered Track', color='blue')
    ax[0].scatter(measurements[:, 2], measurements[:, 0], label='Measurements', color='red', s=10)
    ax[0].plot(zValues[1:], realTrack[:, 0], label='True Track', color='green', linestyle='--')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.7, linewidth=0.5, color='gray')
    
    # y-z Track
    ax[1].set_title("y-z Track")
    ax[1].set_xlabel("z [cm]")
    ax[1].set_ylabel("y [cm]")
    ax[1].plot(zValues, states[:, 1], label='Filtered Track', color='blue')
    ax[1].scatter(measurements[:, 2], measurements[:, 1], label='Measurements', color='red', s=10)
    ax[1].plot(zValues[1:], realTrack[:, 1], label='True Track', color='green', linestyle='--')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.7, linewidth=0.5, color='gray')
    
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=300)

    plt.show()