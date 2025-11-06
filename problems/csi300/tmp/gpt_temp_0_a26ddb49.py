import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price metrics
    df['TrueRange'] = np.maximum(df['high'] - df['low'], 
                                np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
    
    # Mock sector ETF data (in practice would need actual sector ETF data)
    # Using SPY as proxy for sector ETF for demonstration
    spy_high = df['high'].rolling(window=5).mean() * 1.02  # Mock SPY high
    spy_low = df['high'].rolling(window=5).mean() * 0.98   # Mock SPY low
    spy_close = df['close'].rolling(window=5).mean()       # Mock SPY close
    
    # Cross-Asset Volatility Transmission Component
    # Sector Volatility Transmission
    sector_vol_transmission = (spy_high - spy_low) / (df['high'] - df['low'])
    sector_vol_transmission = sector_vol_transmission.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Market Volatility Absorption (using SPY as market proxy)
    market_vol_absorption = (spy_high - spy_low) / (df['high'] - df['low'])
    market_vol_absorption = market_vol_absorption.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Cross-Timezone Volatility Impact (simplified using pre-market range estimate)
    pre_market_high = df['open'] * 1.01  # Mock pre-market high
    pre_market_low = df['open'] * 0.99   # Mock pre-market low
    cross_timezone_impact = (pre_market_high - pre_market_low) / (df['high'] - df['low'])
    cross_timezone_impact = cross_timezone_impact.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volatility Transmission Efficiency
    transmission_lag_effect = (sector_vol_transmission - sector_vol_transmission.shift(1)) / df['TrueRange']
    transmission_lag_effect = transmission_lag_effect.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    absorption_asymmetry = np.sign(market_vol_absorption) * (df['close'] - df['open']) / df['TrueRange']
    absorption_asymmetry = absorption_asymmetry.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    cross_asset_momentum_divergence = (df['close']/df['close'].shift(1)-1) - (spy_close/spy_close.shift(1)-1)
    cross_asset_momentum_divergence = cross_asset_momentum_divergence.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volatility Regime Integration
    transmission_quality_score = transmission_lag_effect * cross_asset_momentum_divergence
    
    # Regime persistence (sign consistency over 3 days)
    sign_consistency = pd.Series(np.where(transmission_lag_effect > 0, 1, -1), index=df.index)
    regime_persistence = sign_consistency.rolling(window=3).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else 0, raw=False
    ) / 3
    
    volatility_context = transmission_quality_score / (market_vol_absorption * sector_vol_transmission)
    volatility_context = volatility_context.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Order Flow Imbalance Dynamics
    # Calculate volume metrics (using full day volume as proxy for intraday periods)
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    
    # Opening Imbalance Momentum
    opening_imbalance_momentum = (df['open'] - df['close'].shift(1)) / (df['volume'] / avg_volume_5d)
    opening_imbalance_momentum = opening_imbalance_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Closing Auction Pressure
    closing_auction_pressure = (df['close'] - df['low']) / (df['volume'] / avg_volume_5d)
    closing_auction_pressure = closing_auction_pressure.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Midday Flow Reversal
    midday_flow_reversal = ((df['high'] + df['low'])/2 - (df['open'] + df['close'])/2) / df['volume']
    midday_flow_reversal = midday_flow_reversal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Order Flow Regime Classification
    aggressive_buying = (opening_imbalance_momentum > 0.02) & (closing_auction_pressure > 0)
    passive_selling = (opening_imbalance_momentum < -0.02) & (closing_auction_pressure < 0)
    balanced_flow = ~(aggressive_buying | passive_selling)
    
    # Flow Quality Assessment
    flow_consistency = np.sign(opening_imbalance_momentum) * np.sign(closing_auction_pressure)
    
    volume_confirmation = (df['volume'] / avg_volume_5d) * flow_consistency
    volume_confirmation = volume_confirmation.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Regime strength (current regime persistence over 3 days)
    regime_strength = pd.Series(0.0, index=df.index)
    for i in range(2, len(df)):
        current_regime = aggressive_buying.iloc[i] if aggressive_buying.iloc[i] else (
            passive_selling.iloc[i] if passive_selling.iloc[i] else balanced_flow.iloc[i]
        )
        count = 0
        for j in range(max(0, i-2), i+1):
            if j < len(df):
                if aggressive_buying.iloc[j] and current_regime == aggressive_buying.iloc[i]:
                    count += 1
                elif passive_selling.iloc[j] and current_regime == passive_selling.iloc[i]:
                    count += 1
                elif balanced_flow.iloc[j] and current_regime == balanced_flow.iloc[i]:
                    count += 1
        regime_strength.iloc[i] = count / 3
    
    # Cross-Asset Flow Integration
    # Determine volatility regime for alignment weights
    stock_volatility = df['high'] - df['low']
    avg_volatility = stock_volatility.rolling(window=20).mean()
    high_vol_regime = stock_volatility > avg_volatility * 1.2
    low_vol_regime = stock_volatility < avg_volatility * 0.8
    
    # Asset-Flow Alignment
    volatility_component = (transmission_quality_score + volatility_context) / 2
    order_flow_component = (flow_consistency + volume_confirmation) / 2
    
    asset_flow_alignment = pd.Series(0.0, index=df.index)
    asset_flow_alignment[high_vol_regime] = 0.6 * volatility_component[high_vol_regime] + 0.4 * order_flow_component[high_vol_regime]
    asset_flow_alignment[low_vol_regime] = 0.4 * volatility_component[low_vol_regime] + 0.6 * order_flow_component[low_vol_regime]
    asset_flow_alignment[~(high_vol_regime | low_vol_regime)] = 0.5 * volatility_component[~(high_vol_regime | low_vol_regime)] + 0.5 * order_flow_component[~(high_vol_regime | low_vol_regime)]
    
    # Order Flow Confirmation Component
    order_flow_confirmation = pd.Series(0.0, index=df.index)
    order_flow_confirmation[aggressive_buying] = flow_consistency[aggressive_buying] * volume_confirmation[aggressive_buying]
    order_flow_confirmation[passive_selling] = -flow_consistency[passive_selling] * volume_confirmation[passive_selling]
    order_flow_confirmation[balanced_flow] = regime_strength[balanced_flow] * midday_flow_reversal[balanced_flow]
    
    # Signal Synthesis
    base_transmission_signal = asset_flow_alignment * transmission_quality_score
    flow_adjustment = base_transmission_signal * order_flow_confirmation
    cross_asset_scaling = flow_adjustment / (market_vol_absorption * sector_vol_transmission)
    cross_asset_scaling = cross_asset_scaling.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Dynamic Alpha Generation
    core_transmission_factor = cross_asset_scaling * transmission_lag_effect
    flow_validation = core_transmission_factor * flow_consistency
    final_alpha = flow_validation * regime_persistence
    
    return final_alpha.fillna(0)
