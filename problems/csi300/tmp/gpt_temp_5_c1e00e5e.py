import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Short-term momentum (5-day)
    data['short_term_momentum'] = (data['close'].shift(1) / data['close'].shift(5) - 1)
    
    # Medium-term momentum (20-day)
    data['medium_term_momentum'] = (data['close'].shift(1) / data['close'].shift(20) - 1)
    
    # Momentum Convergence Assessment
    data['momentum_convergence'] = np.where(
        (data['short_term_momentum'] * data['medium_term_momentum']) > 0,
        np.abs(data['short_term_momentum'] + data['medium_term_momentum']),
        -np.abs(data['short_term_momentum'] - data['medium_term_momentum'])
    )
    
    # Volatility Adjustment - True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['true_range_avg'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    
    # Volume Confirmation
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume_5d_avg'] / data['volume_10d_avg']
    
    # Volume Efficiency Analysis
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Volume Persistence - count consecutive volume surges
    data['volume_surge'] = (data['volume'] > data['volume'].rolling(window=5, min_periods=1).mean()).astype(int)
    data['volume_persistence'] = data['volume_surge'].rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] == 1]) if len(x) > 0 else 0
    )
    
    # Intraday Strength Persistence
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_strength_sign'] = np.sign(data['intraday_strength'])
    
    # Count consecutive same-sign intraday strength
    data['strength_persistence'] = 0
    for i in range(1, len(data)):
        if data['intraday_strength_sign'].iloc[i] == data['intraday_strength_sign'].iloc[i-1]:
            data.loc[data.index[i], 'strength_persistence'] = data['strength_persistence'].iloc[i-1] + 1
    
    # Volume-weighted persistence
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) if len(x) >= 5 else 0
    ).astype(float)
    data['weighted_persistence'] = data['strength_persistence'] * data['volume_percentile']
    
    # Range Efficiency Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_efficiency'] = data['daily_range'] / (data['true_range_avg'] + 1e-8)
    
    # Volume-Volatility Asymmetry
    data['volume_volatility_ratio'] = data['volume'] / (data['true_range_avg'] + 1e-8)
    data['vv_asymmetry'] = (data['volume_volatility_ratio'] - data['volume_volatility_ratio'].rolling(window=10, min_periods=1).mean()) / \
                          (data['volume_volatility_ratio'].rolling(window=10, min_periods=1).std() + 1e-8)
    
    # Price-Volume Trend Efficiency
    data['price_trend'] = data['close'].rolling(window=10, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    data['volume_trend_slope'] = data['volume'].rolling(window=10, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Trend Comparison
    data['trend_alignment'] = np.where(
        (data['price_trend'] * data['volume_trend_slope']) > 0,
        np.abs(data['price_trend'] + data['volume_trend_slope']),
        -np.abs(data['price_trend'] - data['volume_trend_slope'])
    )
    
    # Factor Synthesis - Momentum Efficiency Score
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['momentum_convergence'] / (data['true_range_avg'] + 1e-8)
    
    # Volume confirmation weight
    data['volume_confirmation'] = data['volume_ratio'] * data['volume_persistence']
    
    # Combine components for final alpha factor
    data['momentum_efficiency'] = (
        data['vol_adj_momentum'] * 
        data['volume_confirmation'] * 
        (1 + data['weighted_persistence'] / 10) *
        data['trend_alignment'] *
        data['range_efficiency']
    )
    
    # Reversal Detection
    data['overbought_signal'] = (
        (data['short_term_momentum'] > 0.05) & 
        (data['volume_ratio'] < 0.9) & 
        (data['strength_persistence'] < 2)
    ).astype(int)
    
    data['oversold_signal'] = (
        (data['short_term_momentum'] < -0.05) & 
        (data['volume_ratio'] > 1.1) & 
        (data['strength_persistence'] > 2)
    ).astype(int)
    
    # Final Alpha Generation
    data['alpha_factor'] = np.where(
        data['overbought_signal'] == 1,
        -data['momentum_efficiency'],
        np.where(
            data['oversold_signal'] == 1,
            data['momentum_efficiency'],
            data['momentum_efficiency']
        )
    )
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
