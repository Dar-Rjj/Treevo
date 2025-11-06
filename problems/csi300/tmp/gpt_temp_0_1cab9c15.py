import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Nonlinear Coupling Oscillator Momentum factor that models price-volume interaction
    as coupled oscillators and measures phase synchronization between cycles.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns and volume changes
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Remove NaN values
    data = data.dropna()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Parameters for oscillator analysis
    window_size = 20
    min_periods = 10
    
    for i in range(window_size, len(data)):
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        if len(current_data) < window_size:
            factor_values.iloc[i] = 0
            continue
            
        # Extract recent price and volume data
        price_data = current_data['close'].iloc[-window_size:].values
        volume_data = current_data['volume'].iloc[-window_size:].values
        
        # Normalize the series for oscillator analysis
        price_norm = (price_data - np.mean(price_data)) / (np.std(price_data) + 1e-8)
        volume_norm = (volume_data - np.mean(volume_data)) / (np.std(volume_data) + 1e-8)
        
        # Compute Hilbert transforms to get instantaneous phases
        try:
            # Analytic signals for phase calculation
            price_analytic = signal.hilbert(price_norm)
            volume_analytic = signal.hilbert(volume_norm)
            
            # Instantaneous phases
            price_phase = np.angle(price_analytic)
            volume_phase = np.angle(volume_analytic)
            
            # Phase synchronization measure (phase locking value)
            phase_diff = price_phase - volume_phase
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # Phase coherence (more robust measure)
            phase_coherence = np.abs(np.corrcoef(np.sin(price_phase), np.sin(volume_phase))[0,1])
            
            # Coupling strength persistence (autocorrelation of phase differences)
            if len(phase_diff) > 5:
                coupling_persistence = np.corrcoef(phase_diff[:-1], phase_diff[1:])[0,1]
                coupling_persistence = 0 if np.isnan(coupling_persistence) else coupling_persistence
            else:
                coupling_persistence = 0
            
            # Resonance detection using spectral analysis
            f_price, Pxx_price = signal.periodogram(price_norm)
            f_volume, Pxx_volume = signal.periodogram(volume_norm)
            
            # Find dominant frequencies
            dominant_freq_price = f_price[np.argmax(Pxx_price)]
            dominant_freq_volume = f_volume[np.argmax(Pxx_volume)]
            
            # Frequency matching (resonance)
            freq_match = 1 - np.abs(dominant_freq_price - dominant_freq_volume)
            
            # Combine measures with appropriate weights
            oscillator_momentum = (
                0.4 * plv +                    # Phase locking value
                0.3 * phase_coherence +        # Phase coherence
                0.2 * coupling_persistence +   # Coupling persistence
                0.1 * freq_match               # Frequency resonance
            )
            
            # Apply nonlinear transformation to enhance signal
            oscillator_momentum = np.tanh(oscillator_momentum * 3)
            
            factor_values.iloc[i] = oscillator_momentum
            
        except (ValueError, np.linalg.LinAlgError):
            factor_values.iloc[i] = 0
    
    # Handle any remaining NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
