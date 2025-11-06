import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Volume-Price Efficiency Factor
    Combines price efficiency measures with volume asymmetry patterns
    """
    data = df.copy()
    
    # 1. Measure Intraday Price Efficiency
    # Opening Gap Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['gap_ratio'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['gap_efficiency'] = np.where(
        data['intraday_range'] != 0,
        np.abs(data['gap_ratio']) / data['intraday_range'],
        0
    )
    
    # Closing Efficiency
    data['close_move'] = (data['close'] - data['open']) / data['open']
    data['close_efficiency'] = np.where(
        data['intraday_range'] != 0,
        np.abs(data['close_move']) / data['intraday_range'],
        0
    )
    
    # 2. Quantify Volume Asymmetry
    # Estimate buying/selling pressure using amount and volume
    data['price_change'] = data['close'] - data['open']
    data['vwap'] = data['amount'] / data['volume']
    
    # Upside volume intensity (buying pressure)
    data['upside_volume'] = np.where(
        data['price_change'] > 0,
        data['volume'] * (data['close'] - data['open']) / data['open'],
        0
    )
    
    # Downside volume intensity (selling pressure)
    data['downside_volume'] = np.where(
        data['price_change'] < 0,
        data['volume'] * np.abs(data['close'] - data['open']) / data['open'],
        0
    )
    
    # Volume asymmetry metrics
    data['volume_asymmetry'] = (data['upside_volume'] - data['downside_volume']) / (
        data['upside_volume'] + data['downside_volume'] + 1e-8
    )
    
    # Volume concentration during price moves
    data['volume_concentration'] = np.where(
        data['intraday_range'] != 0,
        data['volume'] / (data['intraday_range'] * data['open'] + 1e-8),
        0
    )
    
    # 3. Combine Efficiency and Asymmetry
    # Confirmed efficiency signals
    data['efficiency_signal'] = data['gap_efficiency'] * data['close_efficiency']
    data['confirmed_efficiency'] = data['efficiency_signal'] * data['volume_asymmetry']
    
    # Divergence warning signals
    data['inefficiency_divergence'] = np.where(
        (data['efficiency_signal'] < data['efficiency_signal'].rolling(5).mean()) &
        (np.abs(data['volume_asymmetry']) > 0.2),
        -np.abs(data['volume_asymmetry']),
        0
    )
    
    # 4. Incorporate Multi-day Persistence
    # Efficiency persistence
    data['efficiency_persistence_3d'] = data['efficiency_signal'].rolling(3).mean()
    data['efficiency_trend_5d'] = data['efficiency_signal'].rolling(5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (np.std(x) + 1e-8)
    )
    
    # Asymmetry persistence
    data['asymmetry_persistence_3d'] = data['volume_asymmetry'].rolling(3).mean()
    data['asymmetry_momentum_5d'] = data['volume_asymmetry'].rolling(5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (np.std(x) + 1e-8)
    )
    
    # 5. Generate Predictive Alpha Factor
    # Composite efficiency-asymmetry indicator
    data['composite_score'] = (
        data['confirmed_efficiency'] * 0.4 +
        data['efficiency_persistence_3d'] * 0.2 +
        data['asymmetry_persistence_3d'] * 0.2 +
        data['inefficiency_divergence'] * 0.2
    )
    
    # Apply temporal smoothing with trend confirmation
    data['smoothed_composite'] = data['composite_score'].rolling(5).mean()
    data['trend_confirmation'] = data['composite_score'].rolling(3).apply(
        lambda x: 1 if all(np.diff(x) > 0) else (-1 if all(np.diff(x) < 0) else 0)
    )
    
    # Final alpha factor
    alpha_factor = data['smoothed_composite'] * (1 + 0.1 * data['trend_confirmation'])
    
    # Clean up and return
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_factor
