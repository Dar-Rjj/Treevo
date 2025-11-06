import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Adaptive Regime Switching
    """
    data = df.copy()
    
    # Price-Volume Divergence Framework
    # Short-term Momentum Divergence
    data['price_momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['volume_momentum_5d'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['divergence_5d'] = data['price_momentum_5d'] - data['volume_momentum_5d']
    
    # Medium-term Trend Confirmation
    data['price_trend_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['volume_trend_20d'] = (data['volume'] - data['volume'].shift(20)) / data['volume'].shift(20)
    data['trend_alignment'] = np.sign(data['price_trend_20d']) * np.sign(data['volume_trend_20d'])
    
    # Divergence Persistence Analysis
    data['divergence_direction'] = np.sign(data['divergence_5d'])
    divergence_persistence = []
    current_streak = 0
    for i in range(len(data)):
        if i == 0:
            divergence_persistence.append(0)
            continue
        if data['divergence_direction'].iloc[i] == data['divergence_direction'].iloc[i-1]:
            current_streak += 1
        else:
            current_streak = 1
        divergence_persistence.append(current_streak)
    data['divergence_persistence'] = divergence_persistence
    
    # Adaptive Regime Detection
    # Volatility Regime Classification
    data['daily_returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['daily_returns'].rolling(window=10, min_periods=5).std()
    data['volatility_60d'] = data['daily_returns'].rolling(window=60, min_periods=30).std()
    data['volatility_regime'] = (data['volatility_10d'] > data['volatility_60d']).astype(int)
    
    # Liquidity Regime Identification
    data['liquidity_efficiency'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_trend_15d'] = data['liquidity_efficiency'].rolling(window=15, min_periods=8).mean()
    data['liquidity_regime'] = (data['liquidity_efficiency'] > data['liquidity_trend_15d']).astype(int)
    
    # Regime-Specific Signal Adjustment
    data['regime_adjusted_divergence'] = data['divergence_5d'].copy()
    high_vol_mask = data['volatility_regime'] == 1
    low_vol_mask = data['volatility_regime'] == 0
    data.loc[high_vol_mask, 'regime_adjusted_divergence'] = data.loc[high_vol_mask, 'divergence_5d'] * 1.5
    data.loc[low_vol_mask, 'regime_adjusted_divergence'] = data.loc[low_vol_mask, 'divergence_5d'] * 0.7
    
    # Intraday Structure Analysis
    # Opening Gap Momentum
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = np.abs(data['close'] - data['open']) / np.abs(data['overnight_gap'] * data['close'].shift(1))
    
    # Midday Reversal Detection
    data['intraday_high_water'] = (data['high'] - data['open']) / data['open']
    data['intraday_low_water'] = (data['open'] - data['low']) / data['open']
    data['failed_breakout'] = ((data['intraday_high_water'] > 0.02) & 
                              (data['close'] < data['open'] * 1.01)).astype(int)
    
    # Closing Auction Pressure
    data['last_hour_move'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volume_vs_avg'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['closing_pressure'] = data['last_hour_move'] * data['volume_vs_avg']
    
    # Cross-Sectional Components (simplified for single stock)
    # Historical Percentile Positioning
    data['divergence_percentile'] = data['divergence_5d'].rolling(window=252, min_periods=126).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) >= 126 else np.nan, raw=False
    )
    
    # Dynamic Signal Integration
    # Component Weight Optimization
    components = ['regime_adjusted_divergence', 'trend_alignment', 'closing_pressure', 
                 'divergence_persistence', 'divergence_percentile']
    
    # Calculate recent performance correlations (simplified)
    weights = []
    for i in range(len(data)):
        if i < 21:
            weights.append([0.2] * 5)  # Equal weights initially
            continue
        
        window_data = data.iloc[max(0, i-21):i]
        if len(window_data) < 10:
            weights.append([0.2] * 5)
            continue
            
        # Simple weight adjustment based on recent volatility-adjusted returns
        component_scores = []
        for comp in components:
            if comp in window_data.columns:
                comp_vals = window_data[comp].dropna()
                if len(comp_vals) > 5:
                    # Use absolute values for signal strength assessment
                    score = np.abs(comp_vals).mean()
                    component_scores.append(score)
                else:
                    component_scores.append(0.1)
            else:
                component_scores.append(0.1)
        
        # Normalize weights
        total_score = sum(component_scores)
        if total_score > 0:
            current_weights = [score/total_score for score in component_scores]
        else:
            current_weights = [0.2] * 5
        weights.append(current_weights)
    
    # Apply weights to components
    final_signal = pd.Series(index=data.index, dtype=float)
    for i, weight_row in enumerate(weights):
        if i < len(data):
            signal_components = []
            for j, comp in enumerate(components):
                if comp in data.columns and not pd.isna(data[comp].iloc[i]):
                    signal_components.append(data[comp].iloc[i] * weight_row[j])
            if signal_components:
                final_signal.iloc[i] = sum(signal_components)
            else:
                final_signal.iloc[i] = 0
    
    # Risk-Adjusted Signal Construction
    volatility_scaling = 1 / (data['volatility_10d'] + 0.01)
    final_signal = final_signal * volatility_scaling
    
    # Portfolio-Level Smoothing
    smoothed_signal = final_signal.rolling(window=3, min_periods=1).mean()
    
    return smoothed_signal
