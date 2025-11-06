import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Biological Market Dynamics Alpha Factor combining neural network propagation,
    cellular metabolism, ecosystem dynamics, genetic evolution, and immune response.
    """
    # Neural Network Price Propagation
    # Synaptic weight adaptation with momentum decay
    momentum_synaptic = (df['close'] / df['close'].shift(5) - 1).rolling(window=10).mean()
    
    # Neural firing rate during volume spikes (normalized volume acceleration)
    volume_accel = df['volume'].pct_change().rolling(window=5).std()
    volume_spike_factor = (df['volume'] / df['volume'].rolling(window=20).mean()) * volume_accel
    
    # Dendritic integration of multi-timeframe signals
    short_ma = df['close'].rolling(window=5).mean()
    medium_ma = df['close'].rolling(window=20).mean()
    long_ma = df['close'].rolling(window=50).mean()
    dendritic_signal = (short_ma / medium_ma) * (medium_ma / long_ma)
    
    # Cellular Market Metabolism
    # Energy expenditure proxy (normalized price range vs volume)
    daily_range = (df['high'] - df['low']) / df['close']
    energy_expenditure = (daily_range * df['volume']) / df['volume'].rolling(window=20).mean()
    
    # Metabolic rate with trading frequency (volume acceleration)
    metabolic_rate = df['volume'].pct_change().rolling(window=5).mean()
    
    # ATP-equivalent cost of large orders (large volume impact)
    large_order_impact = (df['volume'] > df['volume'].rolling(window=20).quantile(0.8)) * \
                        (df['close'] - df['open']) / df['close']
    
    # Ecosystem Population Dynamics
    # Predator-prey in institutional flow (large volume momentum)
    predator_prey = (df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=50).mean()) * \
                   (df['close'].pct_change().rolling(window=5).mean())
    
    # Species diversity (volatility regime diversity)
    vol_short = df['close'].pct_change().rolling(window=5).std()
    vol_medium = df['close'].pct_change().rolling(window=20).std()
    species_diversity = vol_short / vol_medium
    
    # Population density in liquidity (volume concentration)
    volume_density = df['volume'] / df['volume'].rolling(window=20).sum()
    
    # Genetic Algorithm Price Evolution
    # Chromosome crossover in technical patterns
    rsi = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(window=14).mean() / 
                            df['close'].diff().clip(upper=0).abs().rolling(window=14).mean())))
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    chromosome_crossover = (rsi / 50) * (macd / macd.rolling(window=9).mean())
    
    # Mutation rate in volatility changes
    mutation_rate = df['close'].pct_change().rolling(window=10).std().pct_change()
    
    # Fitness selection in trend persistence
    trend_strength = df['close'].rolling(window=10).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    # Immune System Market Response
    # Antibody production to news shocks (abnormal price moves)
    price_shock = (df['close'] - df['close'].shift(1)).abs() / df['close'].rolling(window=20).std()
    antibody_response = price_shock.rolling(window=5).mean()
    
    # Pathogen detection in manipulative trading (abnormal volume-price relationship)
    normal_vp_corr = (df['volume'].rolling(window=20).corr(df['close'].pct_change()))
    current_vp_corr = (df['volume'].rolling(window=5).corr(df['close'].pct_change()))
    pathogen_detection = (current_vp_corr - normal_vp_corr).abs()
    
    # Immune memory in repeated patterns (autocorrelation of returns)
    immune_memory = df['close'].pct_change().rolling(window=10).apply(
        lambda x: pd.Series(x).autocorr() if len(x) > 1 else 0
    )
    
    # Combine all biological components with weighted synthesis
    neural_component = 0.2 * momentum_synaptic + 0.15 * volume_spike_factor + 0.15 * dendritic_signal
    cellular_component = 0.1 * energy_expenditure + 0.1 * metabolic_rate + 0.05 * large_order_impact
    ecosystem_component = 0.08 * predator_prey + 0.07 * species_diversity + 0.05 * volume_density
    genetic_component = 0.06 * chromosome_crossover + 0.04 * mutation_rate + 0.05 * trend_strength
    immune_component = 0.03 * antibody_response + 0.02 * pathogen_detection + 0.01 * immune_memory
    
    # Final biological market dynamics factor
    biological_factor = (neural_component + cellular_component + ecosystem_component + 
                        genetic_component + immune_component)
    
    # Normalize and remove any potential lookahead bias
    biological_factor = (biological_factor - biological_factor.rolling(window=50).mean()) / \
                       biological_factor.rolling(window=50).std()
    
    return biological_factor
