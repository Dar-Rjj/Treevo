import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Multi-Timeframe Volatility Regime Classification
    # Short-term volatility measures
    data['vol_5d'] = data['returns'].rolling(window=5).std()
    data['vol_10d'] = data['returns'].rolling(window=10).std()
    data['volatility_regime'] = data['vol_5d'] / data['vol_10d']
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Range-based volatility confirmation
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    data['atr_20d'] = data['true_range'].rolling(window=20).mean()
    data['range_vol_ratio'] = data['atr_5d'] / data['atr_20d']
    
    # Regime-Adaptive Momentum Components
    # High Volatility Regime Momentum
    data['max_high_5d'] = data['high'].shift(1).rolling(window=5).max()
    data['breakout_momentum'] = (data['close'] - data['max_high_5d']) / data['true_range']
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['high_vol_signal'] = data['breakout_momentum'] * data['gap_momentum'] * data['volatility_regime']
    
    # Low Volatility Regime Momentum
    data['price_change_10d'] = data['close'] - data['close'].shift(10)
    data['abs_returns_10d'] = abs(data['close'] - data['close'].shift(1)).rolling(window=10).sum()
    data['efficiency_ratio'] = data['price_change_10d'] / data['abs_returns_10d']
    data['range_compression'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=10).mean()
    data['low_vol_signal'] = data['efficiency_ratio'] * (1 - data['range_compression'])
    
    # Volume-Volatility Interaction
    # Volume Regime Detection
    data['volume_sma_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_sma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_spike'] = data['volume'] / data['volume_sma_20d']
    data['volume_trend'] = data['volume'] / data['volume_sma_5d']
    data['volume_regime'] = data['volume_spike'] * data['volume_trend']
    
    # Volatility-Volume Alignment
    data['high_vol_volume_align'] = data['high_vol_signal'] * data['volume_regime']
    data['low_vol_volume_align'] = data['low_vol_signal'] * (1 / data['volume_regime'])
    data['regime_volume_score'] = np.where(data['volatility_regime'] > 1, 
                                         data['high_vol_volume_align'], 
                                         data['low_vol_volume_align'])
    
    # Price-Volume Efficiency Metrics
    # Volume-Weighted Price Efficiency
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap_5d_numerator'] = (data['typical_price'] * data['volume']).rolling(window=5).sum()
    data['vwap_5d_denominator'] = data['volume'].rolling(window=5).sum()
    data['vwap_5d'] = data['vwap_5d_numerator'] / data['vwap_5d_denominator']
    
    data['vwap_10d_numerator'] = (data['typical_price'] * data['volume']).rolling(window=10).sum()
    data['vwap_10d_denominator'] = data['volume'].rolling(window=10).sum()
    data['vwap_10d'] = data['vwap_10d_numerator'] / data['vwap_10d_denominator']
    
    data['vwap_efficiency'] = (data['close'] - data['vwap_5d']) / (data['close'] - data['vwap_10d'])
    
    # Amount-Based Confirmation
    data['amount_sma_5d'] = data['amount'].rolling(window=5).mean()
    data['amount_efficiency'] = data['amount'] / data['amount_sma_5d']
    
    # Price-Amount Correlation
    data['price_returns'] = data['close'] / data['close'].shift(1) - 1
    data['amount_returns'] = data['amount'] / data['amount'].shift(1) - 1
    data['price_amount_corr'] = data['price_returns'].rolling(window=10).corr(data['amount_returns'])
    
    data['amount_confirmed_signal'] = data['vwap_efficiency'] * data['amount_efficiency'] * data['price_amount_corr']
    
    # Adaptive Factor Synthesis
    # Regime-Based Signal Selection
    data['high_vol_component'] = data['regime_volume_score'] * data['amount_confirmed_signal']
    data['low_vol_component'] = data['regime_volume_score'] * data['vwap_efficiency']
    data['dynamic_signal'] = np.where(data['volatility_regime'] > 1, 
                                    data['high_vol_component'], 
                                    data['low_vol_component'])
    
    # Trend Confirmation Layer
    data['price_trend'] = np.sign(data['close'] - data['close'].shift(20))
    data['volume_trend_signal'] = np.sign(data['volume'] - data['volume_sma_20d'])
    data['trend_aligned_signal'] = data['dynamic_signal'] * data['price_trend'] * data['volume_trend_signal']
    
    # Final Factor Generation
    # Volatility Normalization
    data['volatility_normalization'] = data['dynamic_signal'] / (data['range_vol_ratio'] * data['volatility_regime'])
    
    # Momentum Persistence
    data['dynamic_signal_lag1'] = data['dynamic_signal'].shift(1)
    data['dynamic_signal_lag2'] = data['dynamic_signal'].shift(2)
    data['dynamic_signal_lag3'] = data['dynamic_signal'].shift(3)
    data['dynamic_signal_lag4'] = data['dynamic_signal'].shift(4)
    data['dynamic_signal_lag5'] = data['dynamic_signal'].shift(5)
    
    # Calculate rolling correlation between lagged dynamic signal and returns
    momentum_persistence = []
    for i in range(len(data)):
        if i >= 5:
            signal_lags = [data['dynamic_signal_lag1'].iloc[i],
                          data['dynamic_signal_lag2'].iloc[i],
                          data['dynamic_signal_lag3'].iloc[i],
                          data['dynamic_signal_lag4'].iloc[i],
                          data['dynamic_signal_lag5'].iloc[i]]
            returns_window = data['price_returns'].iloc[i-4:i+1].values
            if len(signal_lags) == 5 and len(returns_window) == 5:
                corr = np.corrcoef(signal_lags, returns_window)[0, 1]
                momentum_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_persistence.append(0)
        else:
            momentum_persistence.append(0)
    
    data['momentum_persistence'] = momentum_persistence
    
    # Predictive Factor
    data['predictive_factor'] = (data['volatility_normalization'] * 
                               data['momentum_persistence'] * 
                               data['trend_aligned_signal'])
    
    # Return the final factor series
    return data['predictive_factor']
