import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Nonlinear Price-Volume Interaction Topology factor combining:
    - Phase space reconstruction patterns
    - Topological data analysis features
    - Information thermodynamics measures
    """
    data = df.copy()
    
    # Phase space reconstruction parameters
    embedding_dim = 3
    time_delay = 5
    
    # 1. Phase Space Reconstruction Patterns
    def reconstruct_phase_space(series, dim, delay):
        """Delay embedding for phase space reconstruction"""
        embedded = []
        for i in range(len(series) - (dim-1)*delay):
            point = [series.iloc[i + j*delay] for j in range(dim)]
            embedded.append(point)
        return np.array(embedded)
    
    # Normalize price and volume for phase space
    price_norm = (data['close'] - data['close'].rolling(50).mean()) / data['close'].rolling(50).std()
    volume_norm = (data['volume'] - data['volume'].rolling(50).mean()) / data['volume'].rolling(50).std()
    
    # Create combined price-volume state variable
    state_var = price_norm * volume_norm.rolling(10).mean()
    state_var = state_var.fillna(method='ffill')
    
    # 2. Recurrence quantification analysis
    def recurrence_quantification(embedded_data, threshold=0.1):
        """Calculate recurrence rate from phase space embedding"""
        if len(embedded_data) < 2:
            return 0
        dist_matrix = squareform(pdist(embedded_data))
        recurrence_matrix = (dist_matrix <= threshold).astype(float)
        np.fill_diagonal(recurrence_matrix, 0)  # Exclude self-recurrence
        return np.mean(recurrence_matrix)
    
    # 3. Topological persistence features
    def topological_persistence(price_series, window=20):
        """Calculate persistence-like feature from price extremes"""
        if len(price_series) < window:
            return 0
        
        window_data = price_series.iloc[-window:]
        local_max = (window_data == window_data.rolling(5, center=True).max()).sum()
        local_min = (window_data == window_data.rolling(5, center=True).min()).sum()
        
        # Persistence as ratio of local extremes to total points
        persistence = (local_max + local_min) / (2 * len(window_data))
        return persistence
    
    # 4. Information thermodynamics measures
    def market_entropy(price_changes, bins=10):
        """Calculate entropy of price change distribution"""
        if len(price_changes) < bins:
            return 0
        
        # Discretize price changes
        hist, _ = np.histogram(price_changes.dropna(), bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins for entropy calculation
        return entropy(hist)
    
    # Calculate the factor
    factor_values = []
    
    for i in range(len(data)):
        if i < 100:  # Need sufficient history
            factor_values.append(0)
            continue
            
        # Current window data
        window_data = data.iloc[max(0, i-99):i+1]
        
        # Phase space reconstruction
        state_window = state_var.iloc[max(0, i-99):i+1]
        if len(state_window) >= embedding_dim * time_delay:
            embedded = reconstruct_phase_space(state_window, embedding_dim, time_delay)
            recurrence_rate = recurrence_quantification(embedded)
        else:
            recurrence_rate = 0
        
        # Topological persistence
        price_window = data['close'].iloc[max(0, i-49):i+1]
        persistence = topological_persistence(price_window)
        
        # Market entropy
        price_changes = data['close'].iloc[max(0, i-49):i+1].pct_change().dropna()
        market_ent = market_entropy(price_changes)
        
        # Volume-price correlation structure
        vol_price_corr = data['volume'].iloc[max(0, i-19):i+1].corr(
            data['close'].iloc[max(0, i-19):i+1]
        )
        if pd.isna(vol_price_corr):
            vol_price_corr = 0
        
        # Combine components with nonlinear interactions
        # Emphasize states where recurrence is high but entropy is moderate
        # (suggesting structured but not completely predictable behavior)
        nonlinear_interaction = recurrence_rate * (1 - abs(market_ent - 0.5)) * persistence
        
        # Incorporate volume-price correlation structure
        vol_price_structure = abs(vol_price_corr) * (1 if vol_price_corr > 0 else -1)
        
        # Final factor combining topological and thermodynamic aspects
        factor_value = nonlinear_interaction * (1 + 0.5 * vol_price_structure)
        
        factor_values.append(factor_value)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='nonlinear_topology_factor')
    
    # Normalize the factor
    rolling_mean = factor_series.rolling(50, min_periods=1).mean()
    rolling_std = factor_series.rolling(50, min_periods=1).std()
    factor_normalized = (factor_series - rolling_mean) / rolling_std
    
    return factor_normalized.fillna(0)
