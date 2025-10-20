import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum divergence with volume confirmation
    and volatility-adjusted price range signals.
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price Momentum Divergence with Volume Confirmation
    # Calculate Short-Term Momentum (5-day ROC)
    short_term_momentum = (data['close'] / data['close'].shift(5) - 1) * 100
    
    # Calculate Medium-Term Momentum (20-day ROC)
    medium_term_momentum = (data['close'] / data['close'].shift(20) - 1) * 100
    
    # Compute Momentum Divergence
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Apply Volume Filter
    avg_volume_5d = data['volume'].rolling(window=5).mean()
    volume_ratio = data['volume'] / avg_volume_5d
    volume_confirmed_divergence = momentum_divergence * volume_ratio
    
    # Volatility Regime Adjusted Price Range
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate daily returns for volatility calculation
    daily_returns = data['close'].pct_change()
    
    # Determine Volatility Regime
    vol_20d = daily_returns.rolling(window=20).std()
    vol_median = vol_20d.rolling(window=60).median()  # 3-month median volatility
    
    # Adjust Range Signal based on volatility regime
    volatility_regime = vol_20d / vol_median
    range_signal = (data['high'] - data['low']) / data['close']
    
    # Invert in high volatility, amplify in low volatility
    volatility_adjusted_range = np.where(
        volatility_regime > 1.2,
        -range_signal,  # Invert in high volatility
        np.where(
            volatility_regime < 0.8,
            range_signal * 1.5,  # Amplify in low volatility
            range_signal  # Normal in medium volatility
        )
    )
    
    # Combine both signals with appropriate weighting
    # Normalize both components to similar scales
    norm_divergence = (volume_confirmed_divergence - volume_confirmed_divergence.rolling(20).mean()) / volume_confirmed_divergence.rolling(20).std()
    norm_range = (volatility_adjusted_range - volatility_adjusted_range.rolling(20).mean()) / volatility_adjusted_range.rolling(20).std()
    
    # Final factor: 60% momentum divergence, 40% volatility-adjusted range
    final_factor = 0.6 * norm_divergence + 0.4 * norm_range
    
    return final_factor
