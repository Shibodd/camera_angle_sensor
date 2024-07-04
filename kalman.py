import numpy as np

class LinearKalmanFilter:
  def __init__(self, F, H, Q, R, x0, P0):
    """
    Initialize the Kalman Filter.
    
    Parameters:
    F : np.array
        State transition matrix.
    H : np.array
        Observation matrix.
    Q : np.array
        Process noise covariance matrix.
    R : np.array
        Measurement noise covariance matrix.
    x0 : np.array
        Initial state estimate.
    P0 : np.array
        Initial estimate covariance matrix.
    """
    self.F = F
    self.H = H
    self.Q = Q
    self.R = R
    self.x = x0
    self.P = P0

  def predict(self):
    """
    Predict the next state and estimate covariance.
    """
    self.x = self.F @ self.x
    self.P = self.F @ self.P @ self.F.T + self.Q
    return self.x

  def update(self, z, innovation_fn = np.subtract):
    """
    Update the state estimate with a new measurement.
    
    Parameters:
    z : np.array
        The measurement at the current time step.
    """
    # Compute the Kalman gain
    S = self.H @ self.P @ self.H.T + self.R
    K = self.P @ self.H.T @ np.linalg.inv(S)

    # Update the state estimate
    y = innovation_fn(z, self.H @ self.x)
    self.x = self.x + K @ y

    # Update the estimate covariance
    I = np.eye(self.P.shape[0])
    self.P = (I - K @ self.H) @ self.P

  def get_state(self):
    """
    Get the current state estimate.
    
    Returns:
    np.array
        The current state estimate.
    """
    return self.x

  def get_covariance(self):
    """
    Get the current estimate covariance.
    
    Returns:
    np.array
        The current estimate covariance.
    """
    return self.P