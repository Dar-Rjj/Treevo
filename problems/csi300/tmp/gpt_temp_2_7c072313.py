import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Efficiency with Regime-Switching Dynamics
    """
    data = df.copy()
    
    # Calculate basic components
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = data['close'].diff()
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Fractal Efficiency Calculation
    def calculate_fractal_efficiency(window):
        net_change = data['close'].diff(window)
        total_path = data['daily_range'].rolling(window=window).sum()
        efficiency = net_change / total_path
        return efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['eff_5d'] = calculate_fractal_efficiency(5)
    data['eff_10d'] = calculate_fractal_efficiency(10)
    data['eff_20d'] = calculate_fractal_efficiency(20)
    
    # Volume-weighted efficiency
    data['volume_rank'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['vol_weighted_eff'] = data['eff_10d'] * data['volume_rank']
    
    # Multi-timeframe efficiency divergence
    data['eff_divergence'] = data['eff_5d'] - data['eff_20d']
    data['eff_acceleration'] = data['eff_5d'].diff(3)
    
    # Volatility regime classification
    data['atr_10d'] = data['true_range'].rolling(window=10).mean()
    data['volatility_regime'] = pd.cut(
        data['atr_10d'] / data['close'].rolling(window=20).mean(),
        bins=[0, 0.01, 0.03, 1],
        labels=['low', 'medium', 'high']
    )
    
    # Volume distribution analysis
    data['volume_zscore'] = (
        data['volume'] - data['volume'].rolling(window=20).mean()
    ) / data['volume'].rolling(window=20).std()
    data['volume_spike'] = (data['volume_zscore'] > 2).astype(int)
    data['volume_cluster'] = data['volume_spike'].rolling(window=5).sum()
    
    # Regime transition detection
    data['vol_regime_change'] = data['volatility_regime'].ne(data['volatility_regime'].shift(1)).astype(int)
    data['regime_persistence'] = data['vol_regime_change'].rolling(window=10).apply(
        lambda x: 10 - x.sum(), raw=False
    )
    
    # Efficiency-Regime Interaction
    def regime_efficiency_score(row):
        if row['volatility_regime'] == 'low':
            return row['eff_10d'] * 1.2
        elif row['volatility_regime'] == 'medium':
            return row['eff_10d'] * 1.0
        else:  # high volatility
            return row['eff_10d'] * 0.8
    
    data['regime_adj_eff'] = data.apply(regime_efficiency_score, axis=1)
    
    # Volume-efficiency coupling
    data['vol_eff_correlation'] = data['volume'].rolling(window=10).corr(data['eff_10d'])
    data['eff_volume_support'] = np.where(
        (data['vol_eff_correlation'] > 0.3) & (data['volume_rank'] > 0.7),
        data['eff_10d'] * 1.5,
        data['eff_10d'] * 0.5
    )
    
    # Dynamic signal integration
    data['signal_strength'] = (
        data['regime_adj_eff'] * 
        (1 + data['regime_persistence'] / 10) *
        (1 + data['eff_volume_support'])
    )
    
    # Risk-adjusted final signal
    volatility_adj = 1 / (1 + data['atr_10d'] / data['close'])
    volume_confidence = data['volume_rank'] * (1 - data['volume_cluster'] / 5)
    
    data['final_signal'] = (
        data['signal_strength'] * 
        volatility_adj * 
        volume_confidence *
        np.sign(data['eff_divergence'])
    )
    
    # Smooth the final signal
    alpha_factor = data['final_signal'].rolling(window=3).mean()
    
    return alpha_factor
