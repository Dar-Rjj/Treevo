import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Alpha Factor combining Price-Volume Divergence with Regime-Adaptive Reversal
    """
    data = df.copy()
    
    # Price Momentum Component
    data['price_return_5d'] = data['close'].pct_change(5)
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['price_volatility_3d'] = data['high_low_range'].rolling(window=3, min_periods=3).std()
    
    # Volume Momentum Component
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma_5d']
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    # Price-Volume Divergence Calculation
    data['price_volume_divergence'] = data['price_return_5d'] - data['volume_momentum']
    
    # Divergence Persistence
    divergence_sign = np.sign(data['price_volume_divergence'])
    persistence_count = 0
    divergence_persistence = []
    
    for i in range(len(data)):
        if i == 0 or divergence_sign.iloc[i] != divergence_sign.iloc[i-1]:
            persistence_count = 1
        else:
            persistence_count += 1
        divergence_persistence.append(persistence_count)
    
    data['divergence_persistence'] = divergence_persistence
    data['divergence_magnitude'] = data['price_volume_divergence'] * data['divergence_persistence']
    
    # Regime Detection
    # Volatility Regime
    data['returns'] = data['close'].pct_change()
    data['realized_vol_10d'] = data['returns'].rolling(window=10, min_periods=10).std()
    vol_median = data['realized_vol_10d'].rolling(window=20, min_periods=20).median()
    data['high_vol_regime'] = (data['realized_vol_10d'] > vol_median).astype(int)
    
    # Volume Regime
    data['volume_percentile_10d'] = data['volume'].rolling(window=10, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    vol_pct_median = data['volume_percentile_10d'].rolling(window=20, min_periods=20).median()
    data['high_volume_regime'] = (data['volume_percentile_10d'] > vol_pct_median).astype(int)
    
    # Adaptive Reversal Factor Components
    # Short-term mean reversion (2-3 day horizon)
    data['short_term_reversal'] = -data['close'].pct_change(3)
    
    # Momentum breakout signals
    data['momentum_breakout'] = (data['close'] - data['close'].rolling(window=5, min_periods=5).mean()) / data['close'].rolling(window=5, min_periods=5).std()
    
    # Volume confirmation patterns
    data['volume_confirmation'] = np.where(
        np.sign(data['price_return_5d']) == np.sign(data['volume_momentum'] - 1),
        abs(data['price_return_5d']), 
        0
    )
    
    # Regime-Adaptive Signal Combination
    # Base divergence signal
    base_signal = data['divergence_magnitude']
    
    # High Volatility Processing
    high_vol_weight = np.where(data['high_vol_regime'] == 1, 0.7, 0.3)
    high_vol_component = data['short_term_reversal'] * high_vol_weight
    
    # Low Volatility Processing
    low_vol_weight = np.where(data['high_vol_regime'] == 0, 0.6, 0.2)
    low_vol_component = data['momentum_breakout'] * low_vol_weight
    
    # High Volume Processing
    high_volume_weight = np.where(data['high_volume_regime'] == 1, 0.8, 0.2)
    high_volume_component = data['volume_confirmation'] * high_volume_weight
    
    # Low Volume Processing
    low_volume_weight = np.where(data['high_volume_regime'] == 0, 0.4, 0.1)
    low_volume_component = base_signal * low_volume_weight
    
    # Final Alpha Combination
    regime_vol_component = high_vol_component + low_vol_component
    regime_volume_component = high_volume_component + low_volume_component
    
    # Combine all components with regime-specific weights
    final_alpha = (
        0.4 * base_signal +
        0.3 * regime_vol_component +
        0.3 * regime_volume_component
    )
    
    # Normalize the final factor
    alpha_series = final_alpha.rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, 
        raw=False
    )
    
    return alpha_series
