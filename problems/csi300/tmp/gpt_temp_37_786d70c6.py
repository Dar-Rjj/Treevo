import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Reversion with Volume-Pressure Anchoring
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Multi-timeframe volatility assessment
    data['vol_short'] = data['close'].rolling(window=5).std()
    data['vol_medium'] = data['close'].rolling(window=10).std()
    data['vol_long'] = data['close'].rolling(window=20).std()
    
    # Regime classification
    vol_ratio = data['vol_short'] / data['vol_long']
    data['vol_regime'] = np.where(vol_ratio > 1.2, 'high', 
                                 np.where(vol_ratio < 0.8, 'low', 'normal'))
    
    # Momentum Reversion Signal Generation
    # Multi-timeframe momentum calculation
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Reversion pressure signals
    data['rev_short'] = data['mom_3d'] - data['mom_5d']
    data['rev_medium'] = data['mom_5d'] - data['mom_10d']
    data['rev_intensity'] = data['rev_short'] * data['rev_medium']
    
    # Volume-Pressure Anchoring System
    # Volume divergence analysis
    data['vol_sma_5'] = data['volume'].rolling(window=5).mean()
    data['vol_deviation'] = data['volume'] - data['vol_sma_5']
    data['vol_regime_sign'] = np.sign(data['vol_deviation'])
    
    # Intraday pressure signals
    data['range_util'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['gap_persistence'] = data['open'] / data['close'].shift(1) - 1
    
    # Volume-pressure strength
    data['pressure_intensity'] = data['range_util'] * data['gap_persistence']
    data['anchoring_strength'] = data['vol_deviation'] * data['pressure_intensity']
    
    # Liquidity Quality Assessment
    # Volume stability
    data['vol_std_10'] = data['volume'].rolling(window=10).std()
    data['vol_mean_10'] = data['volume'].rolling(window=10).mean()
    data['vol_stability'] = data['vol_std_10'] / (data['vol_mean_10'] + 1e-8)
    
    # Trade efficiency
    data['price_impact'] = (data['high'] - data['low']) / (data['amount'] + 1e-8)
    data['trade_efficiency'] = 1 / (data['price_impact'] + 1e-8)
    
    # Regime-Adaptive Factor Synthesis
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            factor_values.append(0)
            continue
            
        row = data.iloc[i]
        
        if row['vol_regime'] == 'high':
            # High Volatility Regime Processing
            momentum_reversion_adj = row['rev_intensity'] * vol_ratio.iloc[i]
            volume_pressure_conf = row['anchoring_strength'] * np.sign(row['rev_intensity'])
            liquidity_filter = row['vol_stability'] * row['trade_efficiency']
            factor = momentum_reversion_adj * volume_pressure_conf * liquidity_filter
            
        elif row['vol_regime'] == 'low':
            # Low Volatility Regime Processing
            # Calculate ATR (5-day)
            high_low = data['high'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1]
            high_close = abs(data['high'].iloc[i-4:i+1] - data['close'].shift(1).iloc[i-4:i+1])
            low_close = abs(data['low'].iloc[i-4:i+1] - data['close'].shift(1).iloc[i-4:i+1])
            true_ranges = np.maximum(high_low, np.maximum(high_close, low_close))
            atr_5 = true_ranges.mean()
            
            volatility_scaling = atr_5 if atr_5 > 0 else 1e-8
            scaled_reversion = row['rev_intensity'] / volatility_scaling
            anchored_reversion = scaled_reversion * row['anchoring_strength']
            liquidity_quality = row['vol_stability'] * row['trade_efficiency']
            factor = anchored_reversion * liquidity_quality
            
        else:  # Normal volatility
            # Normal Volatility Regime Processing
            base_reversion = row['rev_intensity']
            volume_anchoring = row['anchoring_strength']
            liquidity_assessment = row['vol_stability'] * row['trade_efficiency']
            factor = base_reversion * volume_anchoring * liquidity_assessment
        
        factor_values.append(factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='regime_adaptive_momentum_reversion_anchor')
    
    return factor_series
