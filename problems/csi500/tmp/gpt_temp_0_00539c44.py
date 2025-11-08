import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility State Classification
    # Calculate rolling volatilities
    data['vol_5d'] = data['close'].rolling(window=5).std()
    data['vol_10d'] = data['close'].rolling(window=10).std()
    data['vol_20d'] = data['close'].rolling(window=20).std()
    
    # Volume volatility
    data['volume_vol_5d'] = data['volume'].rolling(window=5).std()
    data['price_volume_vol_ratio'] = data['vol_5d'] / data['volume_vol_5d']
    
    # Volatility regime
    data['volatility_regime'] = data['vol_5d'] / data['vol_10d']
    
    # Regime classification indicators
    data['high_vol_regime'] = (data['volatility_regime'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['volatility_regime'] < 0.8).astype(int)
    data['transition_regime'] = ((data['volatility_regime'] >= 0.8) & (data['volatility_regime'] <= 1.2)).astype(int)
    
    # High volatility patterns
    # Extreme reversal detection
    data['max_drawdown_5d'] = data['close'].rolling(window=5).apply(
        lambda x: (x.min() / x.max()) - 1, raw=True
    )
    data['recovery_potential'] = data['close'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / abs(x.min()), raw=False
    )
    
    # Volatility clustering
    vol_5d_series = data['vol_5d']
    vol_10d_series = data['vol_10d']
    data['consecutive_high_vol_days'] = vol_5d_series.rolling(window=5).apply(
        lambda x: sum(x > vol_10d_series.loc[x.index]), raw=False
    )
    
    # Volatility persistence (simplified)
    data['volatility_persistence'] = vol_5d_series.rolling(window=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, 
        raw=True
    )
    
    # Low volatility patterns
    # Range compression
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    data['range_compression'] = data['daily_range'] / data['avg_range_5d']
    
    # Volume breakout signal
    data['avg_volume_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_breakout_signal'] = data['volume'] / data['avg_volume_10d']
    
    # Momentum accumulation
    data['steady_gain_days'] = data['close'].rolling(window=10).apply(
        lambda x: sum(x.diff().dropna() > 0), raw=False
    )
    
    data['consistent_direction'] = data['close'].rolling(window=10).apply(
        lambda x: np.sign(x.diff().dropna().sum()) if len(x.diff().dropna()) > 0 else 0, 
        raw=False
    )
    
    # Transition regime patterns
    # Volatility expansion detection
    data['volatility_jump'] = data['vol_5d'] / data['vol_20d']
    data['avg_volume_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_expansion'] = data['volume'] / data['avg_volume_20d']
    
    # Regime shift confirmation
    data['price_confirmation'] = (data['close'] - data['close'].shift(5)) / (
        data['close'].shift(5) - data['close'].shift(10) + 1e-8
    )
    data['volume_confirmation'] = data['volume'] / (data['volume'].shift(5) + 1e-8)
    
    # Adaptive Volume Analysis
    # Volume persistence
    volume_series = data['volume']
    data['volume_trend'] = volume_series.rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, 
        raw=True
    )
    
    data['volume_consistency'] = volume_series.rolling(window=5).apply(
        lambda x: x.std() / (x.mean() + 1e-8), raw=True
    )
    
    # Price-volume efficiency
    data['price_movement_per_volume'] = data['daily_range'] / (data['volume'] + 1e-8)
    data['efficiency_trend'] = data['price_movement_per_volume'] / (
        data['price_movement_per_volume'].shift(5) + 1e-8
    )
    
    # Volume impact asymmetry
    up_mask = data['close'] > data['open']
    down_mask = data['close'] < data['open']
    
    data['up_volume_impact'] = np.where(
        up_mask, 
        (data['close'] - data['open']) / (data['volume'] + 1e-8), 
        0
    )
    data['down_volume_impact'] = np.where(
        down_mask, 
        (data['open'] - data['close']) / (data['volume'] + 1e-8), 
        0
    )
    data['volume_impact_asymmetry'] = data['up_volume_impact'] - data['down_volume_impact']
    
    # Adaptive volume signals
    data['high_vol_volume_signal'] = data['volume_consistency'] * data['volume_trend']
    data['low_vol_volume_signal'] = data['volume_breakout_signal'] * data['price_movement_per_volume']
    data['transition_volume_signal'] = data['volume_expansion'] * data['efficiency_trend']
    
    # Regime-Adaptive Alpha Construction
    # High volatility alpha components
    data['reversal_component'] = data['recovery_potential'] * data['volatility_persistence']
    data['volume_component_high'] = data['high_vol_volume_signal'] * data['volume_impact_asymmetry']
    data['high_vol_alpha'] = data['reversal_component'] * data['volume_component_high']
    
    # Low volatility alpha components
    data['breakout_component'] = data['range_compression'] * data['volume_breakout_signal']
    data['momentum_component'] = data['steady_gain_days'] * data['consistent_direction']
    data['low_vol_alpha'] = data['breakout_component'] * data['momentum_component']
    
    # Transition regime alpha components
    data['expansion_component'] = data['volatility_jump'] * data['volume_expansion']
    data['confirmation_component'] = data['price_confirmation'] * data['volume_confirmation']
    data['transition_alpha'] = data['expansion_component'] * data['confirmation_component']
    
    # Final adaptive alpha
    data['adaptive_alpha'] = (
        data['high_vol_regime'] * data['high_vol_alpha'] +
        data['low_vol_regime'] * data['low_vol_alpha'] +
        data['transition_regime'] * data['transition_alpha']
    )
    
    # Clean up and return
    alpha_series = data['adaptive_alpha'].replace([np.inf, -np.inf], np.nan)
    return alpha_series
