import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Gap Momentum with Volume-Price Efficiency factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum Analysis
    # Calculate Gap Magnitude and Extremeness
    data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_percentile'] = data['gap_size'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) >= 10 else 0.5
    )
    
    # Assess Multi-timeframe Momentum Context
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(8) - 1
    
    # Evaluate Momentum Alignment
    data['momentum_alignment'] = np.sign(data['momentum_short']) * np.sign(data['momentum_medium'])
    data['momentum_consistency'] = (abs(data['momentum_short']) + abs(data['momentum_medium'])) / 2
    
    # Volume-Price Efficiency Assessment
    # Intraday Range Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_persistence'] = data['daily_efficiency'].rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(len(x)-1, 1)
    )
    
    # Volume Momentum Confirmation
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(5) - 1
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_breakout'] = data['volume'] / data['avg_volume_10d']
    
    # Price-Volume Relationship Quality
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['price_efficiency'] = (data['close'] - data['close'].shift(1)) / data['true_range'].replace(0, np.nan)
    data['volume_price_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * data['volume']
    
    # Volatility-Regime Context
    # Identify Current Volatility State
    data['returns'] = data['close'].pct_change()
    data['volatility_15d'] = data['returns'].rolling(window=15, min_periods=10).std()
    data['volatility_30d'] = data['returns'].rolling(window=30, min_periods=15).std()
    data['volatility_regime'] = data['volatility_15d'] / data['volatility_30d']
    
    # Regime-Specific Signal Adjustment
    data['regime_adjustment'] = np.where(
        data['volatility_regime'] > 1.2, 0.7,  # High volatility - reduce signal
        np.where(data['volatility_regime'] < 0.8, 1.3, 1.0)  # Low volatility - amplify signal
    )
    
    # Detect Regime Transition Points
    data['volatility_change'] = data['volatility_15d'].pct_change(3)
    data['regime_transition'] = abs(data['volatility_change']) > 0.3
    
    # Composite Signal Generation
    # Core Gap Momentum Factor
    data['core_gap_momentum'] = (
        data['gap_size'] * 
        data['momentum_short'] * 
        data['volume_acceleration'] * 
        data['gap_percentile']
    )
    
    # Efficiency and Quality Enhancement
    data['efficiency_component'] = (
        data['daily_efficiency'] * 
        data['price_efficiency'] * 
        data['volume_breakout'] * 
        np.sign(data['volume_price_alignment'])
    )
    
    # Regime-Adaptive Final Signal
    data['raw_signal'] = (
        data['core_gap_momentum'] * 
        data['efficiency_component'] * 
        data['regime_adjustment'] * 
        data['momentum_alignment']
    )
    
    # Signal Refinement and Risk Adjustment
    # Multi-timeframe Confirmation
    data['momentum_confirmation'] = (
        (data['momentum_short'] * data['momentum_medium'] > 0).astype(int) * 1.2 +
        (data['momentum_short'] * data['momentum_medium'] <= 0).astype(int) * 0.8
    )
    
    # Historical Pattern Integration
    data['gap_success_5d'] = data['gap_size'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x.fillna(0))[0,1] if len(x) >= 3 else 0
    )
    
    # Final Alpha Output with contrarian logic for extreme gaps
    data['final_factor'] = data['raw_signal'] * data['momentum_confirmation'] * (1 + data['gap_success_5d'])
    
    # Apply contrarian logic for extreme gaps (absolute gap size > 5%)
    extreme_gap_mask = abs(data['gap_size']) > 0.05
    data.loc[extreme_gap_mask, 'final_factor'] = -data.loc[extreme_gap_mask, 'final_factor'] * 0.8
    
    # Clean up and return
    result = data['final_factor'].fillna(0)
    return result
