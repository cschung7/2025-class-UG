# AI Instructions: NVDA Regime Analysis with GMM and HMM

## Objective
Download NVDA (Nvidia) daily stock data and perform regime analysis using two different methods:
1. **Gaussian Mixture Model (GMM)** for regime detection
2. **Hidden Markov Model (HMM)** for regime detection

## Task Breakdown

### Step 1: Data Collection
- Download NVDA daily stock data (OHLCV - Open, High, Low, Close, Volume)
- Use a reliable data source (e.g., yfinance, Alpha Vantage, or similar)
- Include sufficient historical data (recommended: at least 5 years for robust regime analysis)
- Store data in a structured format (CSV or pandas DataFrame)

### Step 2: Data Preprocessing
- Calculate returns (log returns or simple returns)
- Calculate additional features that may help identify regimes:
  - Volatility (rolling standard deviation of returns)
  - Volume indicators
  - Price momentum indicators
- Handle missing values appropriately
- Normalize/standardize features if necessary

### Step 3: Gaussian Mixture Model (GMM) Analysis
- Implement GMM to identify market regimes
- Determine optimal number of components (regimes) using:
  - AIC (Akaike Information Criterion)
  - BIC (Bayesian Information Criterion)
  - Cross-validation
- Fit the GMM model to the data
- Assign each observation to a regime based on maximum posterior probability
- Visualize:
  - Regime assignments over time
  - Distribution of returns for each regime
  - Regime transition probabilities
  - Characteristics of each regime (mean returns, volatility, etc.)

### Step 4: Hidden Markov Model (HMM) Analysis
- Implement HMM for regime detection
- Determine optimal number of hidden states (regimes) using similar criteria as GMM
- Fit the HMM model to the data
- Extract:
  - Hidden state sequence (regime assignments)
  - Transition probability matrix
  - Emission probabilities
  - Stationary distribution of states
- Visualize:
  - Regime assignments over time
  - State transition diagram
  - Regime characteristics
  - Comparison with GMM results

### Step 5: Comparison and Analysis
- Compare results from GMM and HMM:
  - Agreement/disagreement in regime identification
  - Regime characteristics (mean, volatility, duration)
  - Transition patterns
- Analyze regime persistence and switching behavior
- Identify periods of high/low volatility regimes
- Calculate regime-specific statistics:
  - Average return per regime
  - Volatility per regime
  - Average regime duration
  - Transition frequencies

### Step 6: Visualization and Reporting
- Create comprehensive visualizations:
  - Time series plot with regime overlays
  - Regime distribution plots
  - Transition probability matrices (heatmaps)
  - Comparison charts between GMM and HMM
- Generate summary statistics and insights
- Document findings and interpretations

## Technical Requirements

### Libraries
- Data: `yfinance`, `pandas`, `numpy`
- GMM: `sklearn.mixture.GaussianMixture` or `scipy.stats`
- HMM: `hmmlearn` or custom implementation
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Statistics: `scipy`, `statsmodels`

### Code Structure
- Modular design with separate functions for:
  - Data downloading
  - Data preprocessing
  - GMM analysis
  - HMM analysis
  - Visualization
  - Comparison and reporting
- Use termcolor for status updates
- Implement proper error handling with try-except blocks
- Use encoding="utf-8" for file operations

### Output Requirements
- Save processed data to CSV
- Save model results (regime assignments, probabilities, etc.)
- Generate and save all visualizations
- Create a summary report with key findings

## Expected Deliverables

1. **Data Files**
   - Raw NVDA daily data (CSV)
   - Processed data with features (CSV)

2. **Model Results**
   - GMM regime assignments and probabilities
   - HMM regime assignments and transition matrices
   - Model parameters and diagnostics

3. **Visualizations**
   - Time series with regime overlays (GMM and HMM)
   - Regime distribution plots
   - Transition probability matrices
   - Comparison charts

4. **Analysis Report**
   - Summary statistics for each regime
   - Comparison between GMM and HMM results
   - Key insights and interpretations
   - Recommendations based on findings

## Notes
- Ensure proper train/validation/test splits if needed
- Consider using log returns for better statistical properties
- Validate model assumptions (normality, stationarity, etc.)
- Document any assumptions or limitations
- Use real data and real calculations - no mock data or artificial results

