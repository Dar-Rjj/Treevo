import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility regime using True Range standard deviation (5-day)
    data['volatility_5d'] = data['true_range'].rolling(window=5).std()
    
    # Historical percentile classification (20-day lookback)
    data['volatility_percentile'] = data['volatility_5d'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Volatility Z-score relative to 20-day history
    data['volatility_zscore'] = data['volatility_5d'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Gap Analysis
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Multi-period return comparison
    data['return_1d'] = data['close'] / data['prev_close'] - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Detect Extreme Gaps (top/bottom 10% of historical distribution)
    data['gap_percentile'] = data['overnight_gap'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    data['extreme_gap'] = ((data['gap_percentile'] >= 0.9) | (data['gap_percentile'] <= 0.1)).astype(int)
    
    # Volume Efficiency & Confirmation
    data['volume_5d_median'] = data['volume'].rolling(window=5).median()
    data['volume_surge'] = data['volume'] / data['volume_5d_median']
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['volume_efficiency'] = data['daily_range'] / (data['volume'] + 1e-8)
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Regime-specific divergence patterns
    data['momentum_divergence'] = np.sign(data['return_1d']) != np.sign(data['return_3d'])
    data['gap_momentum_alignment'] = np.sign(data['overnight_gap']) == np.sign(data['return_1d'])
    
    # Signal Generation & Filtering
    # Regime-specific weighting
    high_vol_regime = (data['volatility_percentile'] > 0.7) & (data['volatility_zscore'] > 1)
    medium_vol_regime = (data['volatility_percentile'] > 0.3) & (data['volatility_percentile'] <= 0.7)
    
    # Base gap reversion signal (negative gap suggests positive return expectation)
    base_reversion = -data['overnight_gap']
    
    # Volume confirmation scoring
    volume_confirmation = (
        (data['volume_surge'] > 1.2) & 
        (data['volume_spike'] > 1.5) & 
        (data['volume_efficiency'] > data['volume_efficiency'].rolling(window=10).median())
    ).astype(int)
    
    # Regime-specific weighting
    regime_weight = np.where(high_vol_regime, 1.5, 
                           np.where(medium_vol_regime, 1.2, 1.0))
    
    # Filter by volatility conditions
    volatility_filter = (data['volatility_zscore'].abs() > 0.5) & (data['extreme_gap'] == 1)
    
    # Combine signals
    divergence_signal = data['momentum_divergence'] & ~data['gap_momentum_alignment']
    
    # Final factor calculation
    factor = (
        base_reversion * 
        regime_weight * 
        volatility_filter.astype(float) * 
        (1 + 0.3 * volume_confirmation) * 
        (1 + 0.2 * divergence_signal.astype(float))
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
