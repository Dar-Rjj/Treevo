import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Efficiency factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['prev_close'])
    data['tr3'] = np.abs(data['low'] - data['prev_close'])
    data['tr'] = np.maximum(data['tr1'], np.maximum(data['tr2'], data['tr3']))
    
    # 2. Volatility Regime Classification
    data['tr_median'] = data['tr'].rolling(window=20, min_periods=10).median()
    data['high_vol_regime'] = data['tr'] > data['tr_median']
    
    # 3. Regime Persistence
    regime_persistence = []
    current_streak = 0
    current_regime = None
    
    for i, regime in enumerate(data['high_vol_regime']):
        if i == 0 or regime != current_regime:
            current_streak = 1
            current_regime = regime
        else:
            current_streak = min(current_streak + 1, 10)  # Cap at 10 days
        regime_persistence.append(current_streak)
    
    data['regime_persistence'] = regime_persistence
    
    # 4. Price-Volume Efficiency Ratio
    data['price_change'] = np.abs(data['close'] - data['prev_close'])
    data['trading_range'] = data['high'] - data['low']
    data['trading_range'] = np.where(data['trading_range'] == 0, 1e-6, data['trading_range'])  # Avoid division by zero
    
    # Directional movement efficiency
    data['directional_efficiency'] = data['price_change'] / data['trading_range']
    data['volume_adjusted_efficiency'] = data['directional_efficiency'] * data['volume']
    
    # Efficiency ratio compared to historical performance
    data['efficiency_ratio'] = data['volume_adjusted_efficiency'].rolling(
        window=20, min_periods=10
    ).apply(lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) * 1 + 
                    (x.iloc[-1] < np.percentile(x.dropna(), 20)) * -1 if len(x.dropna()) > 0 else 0)
    
    # 5. Efficiency Momentum
    data['efficiency_3d_ago'] = data['volume_adjusted_efficiency'].shift(3)
    data['efficiency_momentum'] = data['volume_adjusted_efficiency'] - data['efficiency_3d_ago']
    
    # 6. Volume metrics
    data['volume_prev'] = data['volume'].shift(1)
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_avg_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    
    # 7. Regime-specific signal generation
    signals = []
    
    for i in range(len(data)):
        if i < 10:  # Need enough history
            signals.append(0)
            continue
            
        row = data.iloc[i]
        prev_row = data.iloc[i-1] if i > 0 else None
        
        # High Volatility Regime Signals
        if row['high_vol_regime']:
            # Strong Buy: High Efficiency + Increasing Volume + Persistence
            if (row['efficiency_ratio'] > 0 and 
                row['volume'] > row['volume_prev'] and 
                row['regime_persistence'] >= 3):
                signal_strength = min(row['regime_persistence'] / 7, 1.0)
                if row['efficiency_momentum'] > 0:
                    signal_strength *= 1.2
                signals.append(signal_strength)
            
            # Strong Sell: Low Efficiency + High Volume + Persistence
            elif (row['efficiency_ratio'] < 0 and 
                  row['volume'] > row['volume_avg_5d'] and 
                  row['regime_persistence'] >= 3):
                signal_strength = -min(row['regime_persistence'] / 7, 1.0)
                if row['efficiency_momentum'] < 0:
                    signal_strength *= 1.2
                signals.append(signal_strength)
            
            else:
                signals.append(0)
        
        # Low Volatility Regime Signals
        else:
            # Strong Buy: Moderate Efficiency + Volume Breakout + Recent Change
            if (row['efficiency_ratio'] >= 0 and 
                row['volume'] > 2 * row['volume_avg_10d'] and 
                row['regime_persistence'] <= 2):
                signal_strength = min(row['regime_persistence'] / 7, 1.0)
                if row['efficiency_momentum'] > 0:
                    signal_strength *= 1.2
                signals.append(signal_strength)
            
            # Strong Sell: Low Efficiency + Volume Spike + Recent Change
            elif (row['efficiency_ratio'] < 0 and 
                  row['volume'] > 1.5 * row['volume_avg_10d'] and 
                  row['regime_persistence'] <= 2):
                signal_strength = -min(row['regime_persistence'] / 7, 1.0)
                if row['efficiency_momentum'] < 0:
                    signal_strength *= 1.2
                signals.append(signal_strength)
            
            else:
                signals.append(0)
    
    # Create final factor series
    factor = pd.Series(signals, index=data.index, name='volatility_regime_efficiency')
    
    # Clean up any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
