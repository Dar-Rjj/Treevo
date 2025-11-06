import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price changes
    df['close_change'] = df['close'].diff()
    
    for i in range(8, len(df)):
        current_data = df.iloc[:i+1]
        
        # Cross-Asset Fractal Volatility Framework
        # 5-day upside and downside volatility
        recent_5 = current_data.iloc[-5:]
        upside_vol = np.sqrt(np.sum(np.maximum(0, recent_5['close_change']) ** 2))
        downside_vol = np.sqrt(np.sum(np.maximum(0, -recent_5['close_change']) ** 2))
        volatility_asymmetry_ratio = upside_vol / (downside_vol + 1e-8)
        
        # Multi-scale fractal calculations
        # Short-term (3-day) price fractal
        recent_3 = current_data.iloc[-3:]
        if len(recent_3) >= 3:
            price_range_3 = recent_3['high'].max() - recent_3['low'].min()
            price_variation_3 = np.sum(np.abs(recent_3['close_change']))
            short_term_fractal = np.log(3) / np.log(price_variation_3 / (price_range_3 + 1e-8)) if price_range_3 > 0 else 1.0
        else:
            short_term_fractal = 1.0
        
        # Medium-term (8-day) price fractal
        recent_8 = current_data.iloc[-8:]
        if len(recent_8) >= 8:
            price_range_8 = recent_8['high'].max() - recent_8['low'].min()
            price_variation_8 = np.sum(np.abs(recent_8['close_change']))
            medium_term_fractal = np.log(8) / np.log(price_variation_8 / (price_range_8 + 1e-8)) if price_range_8 > 0 else 1.0
        else:
            medium_term_fractal = 1.0
        
        fractal_divergence = np.abs(short_term_fractal - medium_term_fractal)
        
        # Asymmetric Microstructure Regime Detection
        current_day = current_data.iloc[-1]
        prev_day = current_data.iloc[-2] if i > 0 else current_day
        
        # Intraday volatility efficiency
        high_low_range = current_day['high'] - current_day['low']
        upside_eff = (current_day['high'] - current_day['open']) / (high_low_range + 1e-8)
        downside_eff = (current_day['open'] - current_day['low']) / (high_low_range + 1e-8)
        volatility_skew = upside_eff - downside_eff
        fractal_vol_alignment = volatility_skew * fractal_divergence
        
        # Volume Fractal Analysis
        recent_3_vol = current_data.iloc[-3:]
        if len(recent_3_vol) >= 3:
            volume_range_3 = recent_3_vol['volume'].max() - recent_3_vol['volume'].min()
            volume_variation_3 = np.sum(np.abs(recent_3_vol['volume'].diff().fillna(0)))
            volume_fractal = np.log(3) / np.log(volume_variation_3 / (volume_range_3 + 1e-8)) if volume_range_3 > 0 else 1.0
        else:
            volume_fractal = 1.0
        
        # Volume efficiency and concentration
        volume_efficiency = np.abs(current_day['close'] - prev_day['close']) / (current_day['volume'] + 1e-8)
        recent_4_vol = current_data.iloc[-4:-1]['volume'] if len(current_data) >= 5 else current_data['volume']
        volume_concentration = current_day['volume'] / (np.sum(recent_4_vol) + 1e-8)
        fractal_volume_alignment = volume_fractal * volume_efficiency
        
        # Amount Flow Analysis
        recent_5_amt = current_data.iloc[-5:]
        positive_flow = np.sum([row['amount'] for _, row in recent_5_amt.iterrows() 
                               if row['close'] > current_data.iloc[recent_5_amt.index.get_loc(_)-1]['close']])
        negative_flow = np.sum([row['amount'] for _, row in recent_5_amt.iterrows() 
                               if row['close'] < current_data.iloc[recent_5_amt.index.get_loc(_)-1]['close']])
        fractal_flow_imbalance = ((positive_flow - negative_flow) / (positive_flow + negative_flow + 1e-8)) * volume_fractal
        
        # Opening dynamics
        opening_gap = (current_day['open'] - prev_day['close']) / (prev_day['close'] + 1e-8)
        auction_imbalance = (current_day['open'] - current_day['low']) - (current_day['high'] - current_day['open'])
        opening_efficiency = np.abs(current_day['close'] - current_day['open']) / (np.abs(current_day['open'] - prev_day['close']) + 1e-8)
        fractal_opening_alignment = opening_efficiency * short_term_fractal
        
        # Price Asymmetry and Position
        intraday_bias = (current_day['close'] - current_day['open']) / (high_low_range + 1e-8)
        recent_3_prices = current_data.iloc[-3:]
        fractal_position = (current_day['close'] - recent_3_prices['low'].min()) / (recent_3_prices['high'].max() - recent_3_prices['low'].min() + 1e-8)
        asymmetric_position = fractal_position * intraday_bias
        
        # Gap analysis
        gap_magnitude = np.abs(current_day['open'] / prev_day['close'] - 1)
        gap_filling_efficiency = np.abs(current_day['close'] - current_day['open']) / (np.abs(current_day['open'] - prev_day['close']) + 1e-8)
        fractal_gap_momentum = gap_filling_efficiency * medium_term_fractal
        
        # Support/Resistance with fractal adjustment
        local_support = recent_3_prices['low'].min() * (1 + short_term_fractal / 10)
        local_resistance = recent_3_prices['high'].max() * (1 - short_term_fractal / 10)
        enhanced_position = (current_day['close'] - local_support) / (local_resistance - local_support + 1e-8)
        
        # Fractal regime detection
        if i >= 10:
            prev_short_fractal = np.log(3) / np.log(np.sum(np.abs(current_data.iloc[-5:-2]['close_change'])) / 
                                       (current_data.iloc[-5:-2]['high'].max() - current_data.iloc[-5:-2]['low'].min() + 1e-8))
            prev_medium_fractal = np.log(8) / np.log(np.sum(np.abs(current_data.iloc[-15:-8]['close_change'])) / 
                                        (current_data.iloc[-15:-8]['high'].max() - current_data.iloc[-15:-8]['low'].min() + 1e-8))
            fractal_regime_shift = (short_term_fractal / (prev_short_fractal + 1e-8)) - (medium_term_fractal / (prev_medium_fractal + 1e-8))
        else:
            fractal_regime_shift = 0
        
        fractal_anchor_convergence = enhanced_position * fractal_regime_shift
        
        # Range compression analysis
        if i >= 5:
            current_range = high_low_range
            prev_range_4 = current_data.iloc[-5]['high'] - current_data.iloc[-5]['low']
            range_compression = current_range / (prev_range_4 + 1e-8)
        else:
            range_compression = 1.0
        
        # Compression duration estimation (simplified)
        compression_duration = 1 if range_compression < 0.8 and fractal_divergence < 0.1 else 0
        fractal_compression_exhaustion = compression_duration * range_compression * fractal_divergence
        
        # Regime-adaptive alpha synthesis
        # Determine regime based on fractal characteristics
        if fractal_divergence > 0.15 and np.abs(fractal_vol_alignment) > 0.1:
            # High Fractal Volatility Regime
            alpha_component = (volatility_asymmetry_ratio * 0.25 + 
                             fractal_vol_alignment * 0.18 + 
                             asymmetric_position * 0.08)
        elif range_compression < 0.8 and fractal_divergence < 0.1:
            # Low Fractal Compression Regime
            alpha_component = (fractal_compression_exhaustion * 0.3 + 
                             fractal_volume_alignment * 0.15 + 
                             fractal_anchor_convergence * 0.05)
        else:
            # Transition Fractal Regime
            alpha_component = (fractal_regime_shift * 0.25 + 
                             fractal_flow_imbalance * 0.12 + 
                             fractal_opening_alignment * 0.10)
        
        # Additional cross-asset components (simulated)
        cross_asset_momentum = fractal_divergence * 0.20
        gap_momentum_component = fractal_gap_momentum * 0.07
        
        # Final alpha calculation
        final_alpha = (alpha_component + cross_asset_momentum + gap_momentum_component)
        
        # Normalize to reasonable range
        alpha.iloc[i] = np.tanh(final_alpha)
    
    # Fill initial NaN values
    alpha = alpha.fillna(0)
    
    return alpha
