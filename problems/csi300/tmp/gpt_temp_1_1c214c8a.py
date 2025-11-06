import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on cross-market microstructure patterns and temporal dynamics
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Feature 1: Cross-Market Microstructure Contagion
    # Calculate rolling correlation between volume and price movement as proxy for order book pressure
    volume_returns_corr = data['volume'].rolling(window=20).corr(data['close'].pct_change())
    
    # Feature 2: Behavioral Anchoring Deviation  
    # Measure distance from recent high/low psychological levels with volume confirmation
    recent_high = data['high'].rolling(window=10).max()
    recent_low = data['low'].rolling(window=10).min()
    price_midpoint = (data['high'] + data['low']) / 2
    
    # Anchoring strength based on volume concentration near psychological levels
    high_anchor_strength = (data['volume'] * (1 - abs(data['high'] - recent_high) / recent_high)).rolling(window=5).mean()
    low_anchor_strength = (data['volume'] * (1 - abs(data['low'] - recent_low) / recent_low)).rolling(window=5).mean()
    anchoring_deviation = (high_anchor_strength + low_anchor_strength) / 2
    
    # Feature 3: Temporal Asymmetry in Price Discovery
    # Analyze which parts of the trading range contribute most to price formation
    open_to_close_range = abs(data['close'] - data['open'])
    daily_range = data['high'] - data['low']
    price_discovery_efficiency = (open_to_close_range / daily_range).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Rolling asymmetry in price discovery efficiency
    discovery_asymmetry = price_discovery_efficiency.rolling(window=10).std()
    
    # Feature 4: Microstructural Memory Persistence
    # Calculate autocorrelation of order flow patterns (using volume and amount as proxies)
    volume_autocorr = data['volume'].rolling(window=15).apply(lambda x: x.autocorr(), raw=False)
    amount_autocorr = data['amount'].rolling(window=15).apply(lambda x: x.autocorr(), raw=False)
    memory_persistence = (volume_autocorr + amount_autocorr) / 2
    
    # Feature 5: Information Diffusion Velocity
    # Measure speed of price adjustment using intraday volatility patterns
    intraday_volatility = (data['high'] - data['low']) / data['open']
    diffusion_velocity = intraday_volatility.rolling(window=5).std()
    
    # Combine features with appropriate weights
    factor = (
        0.3 * volume_returns_corr.fillna(0) +
        0.25 * anchoring_deviation.fillna(0) +
        0.2 * discovery_asymmetry.fillna(0) +
        0.15 * memory_persistence.fillna(0) +
        0.1 * diffusion_velocity.fillna(0)
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor.fillna(0)
