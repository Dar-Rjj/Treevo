import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Asymmetry Efficiency Momentum with Gap-Volume Confirmation
    """
    data = df.copy()
    
    # 1. Multi-Timeframe Volatility Asymmetry Components
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Calculate upside and downside volatility components
    data['upside_move'] = np.where(data['high'] > data['open'], data['high'] - data['open'], 0)
    data['downside_move'] = np.where(data['low'] < data['open'], data['open'] - data['low'], 0)
    
    # Rolling volatility asymmetry calculation
    def calc_asymmetry(window):
        positive_days = window['returns'] > 0
        negative_days = window['returns'] < 0
        
        if positive_days.sum() > 0 and negative_days.sum() > 0:
            upside_vol = window.loc[positive_days, 'upside_move'].mean()
            downside_vol = window.loc[negative_days, 'downside_move'].mean()
            return upside_vol / downside_vol if downside_vol > 0 else 1.0
        return 1.0
    
    # Calculate 10-day volatility asymmetry
    asymmetry_values = []
    for i in range(len(data)):
        if i >= 10:
            window = data.iloc[i-9:i+1].copy()
            asymmetry_values.append(calc_asymmetry(window))
        else:
            asymmetry_values.append(1.0)
    
    data['volatility_asymmetry'] = asymmetry_values
    
    # Multi-timeframe asymmetry momentum
    data['asymmetry_momentum_5d'] = data['volatility_asymmetry'] - data['volatility_asymmetry'].shift(5)
    data['asymmetry_momentum_20d'] = data['volatility_asymmetry'] - data['volatility_asymmetry'].shift(20)
    
    # 2. Gap-Enhanced Efficiency Divergence
    # Fractal Efficiency Calculation
    data['net_movement'] = data['close'] - data['close'].shift(10)
    data['total_movement'] = data['close'].diff().abs().rolling(window=10).sum()
    data['fractal_efficiency'] = data['net_movement'] / np.where(data['total_movement'] > 0, data['total_movement'], 1)
    
    # Gap Momentum
    data['gap_momentum'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    
    # Gap-Weighted Efficiency Divergence
    data['gap_weighted_divergence'] = (data['asymmetry_momentum_5d'] - data['asymmetry_momentum_20d']) * data['gap_momentum'] * data['fractal_efficiency']
    
    # 3. Volume-Volatility Integration
    # Volume Efficiency Dynamics
    data['volume_efficiency'] = data['volume'] / np.where(data['high'] - data['low'] > 0, data['high'] - data['low'], 1)
    data['volume_efficiency_momentum'] = data['volume_efficiency'] - data['volume_efficiency'].shift(5)
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Volume-Asymmetry Alignment
    def rolling_correlation(x, y, window):
        correlations = []
        for i in range(len(x)):
            if i >= window:
                x_window = x.iloc[i-window+1:i+1]
                y_window = y.iloc[i-window+1:i+1]
                if len(x_window) == window and len(y_window) == window:
                    corr = x_window.corr(y_window)
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
        return correlations
    
    data['volume_asymmetry_corr'] = rolling_correlation(data['volume'], data['volatility_asymmetry'], 8)
    
    # Amount-Based Liquidity Scaling
    data['liquidity_scaling'] = data['amount'] / data['amount'].rolling(window=10).median()
    
    # 4. Volatility Regime Adaptive Weighting
    # Intraday Volatility Regime
    data['current_volatility'] = (data['high'] - data['low']) / data['close']
    data['atr_10d'] = (data['high'] - data['low']).rolling(window=10).mean() / data['close']
    data['atr_20d'] = (data['high'] - data['low']).rolling(window=20).mean() / data['close']
    data['volatility_persistence'] = data['atr_10d'] / data['atr_20d']
    
    # Efficiency Regime Analysis
    data['efficiency_regime'] = np.where(data['fractal_efficiency'].abs() > 0.5, 1.2,  # High efficiency
                                np.where(data['fractal_efficiency'].abs() < 0.2, 0.8,   # Low efficiency
                                         1.0))  # Normal efficiency
    
    # Regime-Specific Multipliers
    data['regime_multiplier'] = np.where(data['volatility_persistence'] > 1.1, 1.3,  # High volatility persistence
                                np.where(data['volatility_persistence'] < 0.9, 0.7,   # Low volatility persistence
                                         1.0))  # Normal regime
    
    # 5. Composite Alpha Generation
    # Core Signal
    data['core_signal'] = data['gap_weighted_divergence'] * data['volume_confirmation']
    
    # Volume-Asymmetry Alignment Enhancement
    data['alignment_enhancement'] = np.where(data['volume_asymmetry_corr'] > 0.5, 1.5,  # Strong alignment
                                    np.where(data['volume_asymmetry_corr'] < -0.5, 0.5,  # Diverging patterns
                                             1.0))  # Weak alignment
    
    # Regime-Adjusted Signal
    data['regime_adjusted_signal'] = data['core_signal'] * data['regime_multiplier'] * data['liquidity_scaling'] * data['alignment_enhancement']
    
    # Final Alpha
    data['alpha'] = data['regime_adjusted_signal'] * data['volume_efficiency_momentum'] * data['efficiency_regime']
    
    # Clean up and return
    alpha_series = data['alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
