import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate rolling averages for efficiency
    df['range_4d'] = df['high'] - df['low']
    df['range_19d'] = df['high'] - df['low']
    df['vol_avg_4d'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['amount_avg_4d'] = df['amount'].rolling(window=5, min_periods=1).mean()
    df['range_avg_4d'] = df['range_4d'].rolling(window=5, min_periods=1).mean()
    df['range_avg_19d'] = df['range_19d'].rolling(window=20, min_periods=1).mean()
    df['range_avg_8d'] = df['range_4d'].rolling(window=9, min_periods=1).mean()
    
    for i in range(len(df)):
        if i < 21:  # Need sufficient history
            alpha.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_data = df.iloc[:i+1]  # Only use current and past data
        
        # Fractal Momentum Dynamics
        # Velocity Divergence
        short_medium = (current['close'] / df.iloc[i-3]['close'] - 1) - (current['close'] / df.iloc[i-8]['close'] - 1)
        medium_long = (current['close'] / df.iloc[i-8]['close'] - 1) - (current['close'] / df.iloc[i-21]['close'] - 1)
        acceleration_spread = abs(short_medium) * abs(medium_long)
        
        # Range Fractality
        short_range_scaling = current['range_4d'] / prev_data['range_avg_4d'].iloc[i]
        medium_range_scaling = current['range_4d'] / prev_data['range_avg_19d'].iloc[i]
        range_fractality_ratio = short_range_scaling / medium_range_scaling if medium_range_scaling != 0 else 0
        
        # Volume-Enhanced Momentum
        volume_momentum = (current['volume'] / df.iloc[i-5]['volume'] - 1) - (current['volume'] / df.iloc[i-10]['volume'] - 1)
        volume_price_alignment = volume_momentum * short_medium
        
        # Asymmetric Microstructure Dynamics
        # Order Flow Asymmetry
        opening_order_flow = (current['open'] - df.iloc[i-1]['close']) * (current['amount'] / current['volume'] if current['volume'] != 0 else 0)
        closing_order_pressure = (current['close'] - (current['high'] + current['low'])/2) * ((current['amount'] - df.iloc[i-1]['amount']) / df.iloc[i-1]['amount'] if df.iloc[i-1]['amount'] != 0 else 0)
        order_flow_asymmetry = opening_order_flow - closing_order_pressure
        
        # Volatility Asymmetry
        gap_momentum = (current['open'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close'] if df.iloc[i-1]['close'] != 0 else 0
        intraday_efficiency = (current['close'] - current['open']) / current['range_4d'] if current['range_4d'] != 0 else 0
        vol_asymmetry = current['range_4d'] / abs(current['open'] - df.iloc[i-1]['close']) if abs(current['open'] - df.iloc[i-1]['close']) != 0 else 0
        
        # Volume-Price Dynamics
        volume_scaling = current['volume'] / prev_data['vol_avg_4d'].iloc[i] if prev_data['vol_avg_4d'].iloc[i] != 0 else 0
        amount_scaling = current['amount'] / prev_data['amount_avg_4d'].iloc[i] if prev_data['amount_avg_4d'].iloc[i] != 0 else 0
        amount_volume_coordination = np.sign(volume_scaling - 1) * np.sign(amount_scaling - 1)
        volume_price_imbalance = (current['close'] / df.iloc[i-1]['close'] - 1) - (current['volume'] / df.iloc[i-1]['volume'] - 1)
        
        # Structural Break & Regime Adaptation
        price_break_intensity = abs(current['close'] - df.iloc[i-1]['close']) / (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) if (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) != 0 else 0
        volume_confirmation = volume_scaling
        
        # Volatility Regime Framework
        range_regime = current['range_4d'] / prev_data['range_avg_8d'].iloc[i] if prev_data['range_avg_8d'].iloc[i] != 0 else 0
        
        # Calculate volatility ratios
        vol_ratio_4d = prev_data['range_4d'].iloc[i-4:i+1].mean() / prev_data['close'].iloc[i-4:i+1].mean() if prev_data['close'].iloc[i-4:i+1].mean() != 0 else 0
        vol_ratio_19d = prev_data['range_4d'].iloc[i-19:i+1].mean() / prev_data['close'].iloc[i-19:i+1].mean() if prev_data['close'].iloc[i-19:i+1].mean() != 0 else 0
        volatility_regime = vol_ratio_4d / vol_ratio_19d if vol_ratio_19d != 0 else 0
        
        # Flow Regime Adaptation
        flow_regime = (1 if (volume_scaling > 1.2 and amount_scaling > 1.1) else 0) - (1 if (volume_scaling < 0.8 and amount_scaling < 0.9) else 0)
        
        # Multi-Scale Fractal Integration
        micro_scale = intraday_efficiency
        meso_scale = (current['close'] - df.iloc[i-2]['close']) / (prev_data['high'].iloc[i-2:i+1].max() - prev_data['low'].iloc[i-2:i+1].min()) if (prev_data['high'].iloc[i-2:i+1].max() - prev_data['low'].iloc[i-2:i+1].min()) != 0 else 0
        macro_scale = (current['close'] - df.iloc[i-5]['close']) / (prev_data['high'].iloc[i-5:i+1].max() - prev_data['low'].iloc[i-5:i+1].min()) if (prev_data['high'].iloc[i-5:i+1].max() - prev_data['low'].iloc[i-5:i+1].min()) != 0 else 0
        fractal_fusion = micro_scale * meso_scale * macro_scale
        
        # Efficiency & Liquidity Enhancement
        gap_efficiency = ((current['open'] - df.iloc[i-1]['close']) / (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) if (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) != 0 else 0) * intraday_efficiency
        price_impact = intraday_efficiency * (current['amount'] / current['volume'] if current['volume'] != 0 else 0)
        flow_consistency = np.sign(volume_scaling) * np.sign(amount_scaling)
        
        volume_concentration = current['volume'] / current['range_4d'] if current['range_4d'] != 0 else 0
        liquidity_flow = current['volume'] / df.iloc[i-5]['volume'] - 1 if df.iloc[i-5]['volume'] != 0 else 0
        volume_weighted_reversal = current['volume'] * (current['close'] - df.iloc[i-1]['close']) / current['range_4d'] if current['range_4d'] != 0 else 0
        
        # Dynamic Alpha Synthesis
        # Core components
        regime_integration = flow_regime * range_regime
        break_signal = price_break_intensity * volume_confirmation
        momentum_core = fractal_fusion * volume_price_alignment
        
        # Volatility adjustment
        dynamic_volatility = np.sqrt(vol_asymmetry + 1e-8)
        volatility_adjusted_momentum = momentum_core / dynamic_volatility
        
        # Liquidity enhancement
        confidence_weighting = flow_consistency * amount_volume_coordination
        liquidity_enhanced_momentum = volatility_adjusted_momentum * confidence_weighting
        
        # Regime-based signal selection
        expansion_signal = acceleration_spread * range_regime
        contraction_signal = volume_weighted_reversal * intraday_efficiency
        
        # Final alpha composition
        if flow_regime >= 0:  # Expansion phase
            regime_signal = expansion_signal
        else:  # Contraction phase
            regime_signal = contraction_signal
        
        efficiency_enhancement = gap_efficiency * price_impact
        
        final_alpha = (liquidity_enhanced_momentum * regime_signal * efficiency_enhancement * 
                      regime_integration * break_signal * vol_asymmetry)
        
        alpha.iloc[i] = final_alpha
    
    # Clean infinite and NaN values
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha
