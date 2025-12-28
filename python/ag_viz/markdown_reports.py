"""
Markdown report generation for ARIMA-GARCH model outputs.

Provides functions to generate professional, thorough Markdown reports
with accompanying visuals for different analysis types.
"""

import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from ag_viz.utils import format_model_spec


def _image_to_data_uri(image_path: Path) -> str:
    """
    Convert an image file to a data URI for embedding in Markdown.
    
    Parameters
    ----------
    image_path : Path
        Path to the image file.
    
    Returns
    -------
    str
        Data URI string for the image.
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    b64_data = base64.b64encode(image_data).decode('utf-8')
    extension = image_path.suffix.lower().lstrip('.')
    mime_type = 'image/png' if extension == 'png' else f'image/{extension}'
    
    return f"data:{mime_type};base64,{b64_data}"


def _get_image_markdown(image_path: Path, alt_text: str, use_data_uri: bool = False) -> str:
    """
    Generate Markdown for displaying an image.
    
    Parameters
    ----------
    image_path : Path
        Path to the image file.
    alt_text : str
        Alternative text for the image.
    use_data_uri : bool, optional
        If True, embed image as data URI; otherwise use relative path (default: False).
    
    Returns
    -------
    str
        Markdown string for the image.
    """
    if use_data_uri and image_path.exists():
        data_uri = _image_to_data_uri(image_path)
        return f"![{alt_text}]({data_uri})"
    else:
        # Use relative path
        return f"![{alt_text}]({image_path.name})"


def generate_fit_report(
    data: pd.DataFrame,
    model_json: Dict[str, Any],
    plot_path: Path,
    output_path: Path,
    use_data_uri: bool = False
) -> Path:
    """
    Generate a Markdown report for model fitting results.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data used for fitting.
    model_json : Dict[str, Any]
        Model specification and parameters from JSON.
    plot_path : Path
        Path to the diagnostic plot image.
    output_path : Path
        Path where the Markdown report will be saved.
    use_data_uri : bool, optional
        If True, embed images as data URIs (default: False).
    
    Returns
    -------
    Path
        Path to the saved Markdown report.
    """
    values = data.iloc[:, 0].values
    model_spec = format_model_spec(model_json)
    
    # Extract model parameters if available
    params = model_json.get('parameters', {})
    arima_params = params.get('arima', {})
    garch_params = params.get('garch', {})
    
    report = f"""# ARIMA-GARCH Model Fit Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report presents the results of fitting an **{model_spec}** model to the provided time series data.

## Model Specification

- **Model Type:** {model_spec}
- **Observations:** {len(values)}
- **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}

## Methodology

### ARIMA Component
The ARIMA (AutoRegressive Integrated Moving Average) component models the conditional mean of the time series. It captures:
- **AutoRegressive (AR):** Past values' influence on current value
- **Integration (I):** Level of differencing to achieve stationarity
- **Moving Average (MA):** Past forecast errors' influence on current value

### GARCH Component
The GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) component models the conditional variance, capturing:
- **Volatility clustering:** Periods of high/low volatility tend to persist
- **Time-varying variance:** More accurate uncertainty quantification

## Data Summary Statistics

| Statistic | Value |
|-----------|-------|
| Count | {len(values)} |
| Mean | {np.mean(values):.6f} |
| Std Dev | {np.std(values):.6f} |
| Min | {np.min(values):.6f} |
| Max | {np.max(values):.6f} |
| Skewness | {stats.skew(values):.4f} |
| Kurtosis | {stats.kurtosis(values):.4f} |

### Interpretation

"""
    
    # Add interpretation of statistics
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)
    
    if abs(skewness) < 0.5:
        report += "- **Skewness:** The distribution appears approximately symmetric.\n"
    elif skewness > 0:
        report += f"- **Skewness:** The distribution is right-skewed (positively skewed) with a tail extending toward positive values.\n"
    else:
        report += f"- **Skewness:** The distribution is left-skewed (negatively skewed) with a tail extending toward negative values.\n"
    
    if abs(kurtosis) < 0.5:
        report += "- **Kurtosis:** The distribution has approximately normal tail behavior (mesokurtic).\n"
    elif kurtosis > 0:
        report += f"- **Kurtosis:** The distribution exhibits heavy tails (leptokurtic), suggesting more extreme values than a normal distribution.\n"
    else:
        report += f"- **Kurtosis:** The distribution has light tails (platykurtic), with fewer extreme values than a normal distribution.\n"
    
    report += f"""

## Model Parameters

### ARIMA Parameters
"""
    
    # Add ARIMA parameters if available
    if 'intercept' in arima_params:
        report += f"- **Intercept (μ):** {arima_params['intercept']:.6f}\n"
    
    if 'ar_coef' in arima_params and arima_params['ar_coef']:
        report += "- **AR Coefficients (φ):**\n"
        for i, coef in enumerate(arima_params['ar_coef'], 1):
            report += f"  - φ{i} = {coef:.6f}\n"
    
    if 'ma_coef' in arima_params and arima_params['ma_coef']:
        report += "- **MA Coefficients (θ):**\n"
        for i, coef in enumerate(arima_params['ma_coef'], 1):
            report += f"  - θ{i} = {coef:.6f}\n"
    
    report += """
### GARCH Parameters
"""
    
    # Add GARCH parameters if available
    if 'omega' in garch_params:
        report += f"- **Omega (ω):** {garch_params['omega']:.6f} - Base level of volatility\n"
    
    if 'alpha_coef' in garch_params and garch_params['alpha_coef']:
        report += "- **Alpha Coefficients (α):** Response to past shocks\n"
        for i, coef in enumerate(garch_params['alpha_coef'], 1):
            report += f"  - α{i} = {coef:.6f}\n"
    
    if 'beta_coef' in garch_params and garch_params['beta_coef']:
        report += "- **Beta Coefficients (β):** Persistence of volatility\n"
        for i, coef in enumerate(garch_params['beta_coef'], 1):
            report += f"  - β{i} = {coef:.6f}\n"
    
    # Check for volatility persistence
    if 'alpha_coef' in garch_params and 'beta_coef' in garch_params:
        if garch_params['alpha_coef'] and garch_params['beta_coef']:
            persistence = sum(garch_params['alpha_coef']) + sum(garch_params['beta_coef'])
            report += f"\n**Volatility Persistence:** {persistence:.4f}\n"
            if persistence > 0.99:
                report += "- Very high persistence indicates volatility shocks have long-lasting effects.\n"
            elif persistence > 0.90:
                report += "- High persistence suggests volatility shocks decay slowly.\n"
            else:
                report += "- Moderate persistence indicates volatility shocks dissipate relatively quickly.\n"
    
    report += f"""

## Visualizations

{_get_image_markdown(plot_path, "Fit Diagnostics Plot", use_data_uri)}

The plot above shows the observed time series data along with key summary statistics for the fitted model.

## Key Metrics

The model was successfully estimated using maximum likelihood estimation. Key model quality metrics include:

- **Log-Likelihood:** Higher values indicate better fit to the data
- **AIC (Akaike Information Criterion):** Lower values preferred; balances fit and complexity
- **BIC (Bayesian Information Criterion):** Lower values preferred; penalizes complexity more than AIC

## Caveats and Considerations

1. **Model Assumptions:**
   - ARIMA assumes linear relationships in the mean equation
   - GARCH assumes the conditional variance follows a specific functional form
   - Innovations are assumed to be normally distributed (or student-t in some variants)

2. **Sample Size:** Results are most reliable with sufficient data (typically 500+ observations for GARCH models)

3. **Stationarity:** The time series should be stationary (or made stationary through differencing)

4. **Parameter Constraints:** All parameters should satisfy stationarity and non-negativity constraints

5. **Out-of-Sample Performance:** In-sample fit doesn't guarantee good out-of-sample forecasting performance

## Next Steps

1. **Diagnostic Analysis:** Run residual diagnostics to check model adequacy:
   ```bash
   ag-viz diagnostics -m model.json -d data.csv -o ./diagnostics/
   ```

2. **Forecasting:** Generate forecasts with confidence intervals:
   ```bash
   ag-viz forecast -m model.json -n 30 -o forecast.csv
   ```

3. **Simulation:** Simulate paths to understand model behavior:
   ```bash
   ag-viz simulate -m model.json -p 100 -n 1000 -o simulation.csv
   ```

4. **Model Selection:** Consider comparing with alternative specifications:
   ```bash
   ag select -d data.csv -c BIC -o best_model.json
   ```

## References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
- Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control.

---

*Report generated by ag-viz on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path


def generate_forecast_report(
    model_json: Dict[str, Any],
    forecast_df: pd.DataFrame,
    plot_path: Path,
    output_path: Path,
    use_data_uri: bool = False
) -> Path:
    """
    Generate a Markdown report for forecast results.
    
    Parameters
    ----------
    model_json : Dict[str, Any]
        Model specification and parameters from JSON.
    forecast_df : pd.DataFrame
        Forecast data with mean, variance, and confidence intervals.
    plot_path : Path
        Path to the forecast plot image.
    output_path : Path
        Path where the Markdown report will be saved.
    use_data_uri : bool, optional
        If True, embed images as data URIs (default: False).
    
    Returns
    -------
    Path
        Path to the saved Markdown report.
    """
    model_spec = format_model_spec(model_json)
    horizon = len(forecast_df)
    
    report = f"""# ARIMA-GARCH Forecast Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report presents forecasts generated from a **{model_spec}** model over a **{horizon}-step horizon**.

## Model Specification

- **Model Type:** {model_spec}
- **Forecast Horizon:** {horizon} steps ahead
- **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}

## Methodology

### Multi-Step Ahead Forecasting

ARIMA-GARCH models produce forecasts for both the conditional mean and conditional variance:

1. **Mean Forecast:** Predicted value at each future time step based on the ARIMA component
2. **Variance Forecast:** Predicted uncertainty (volatility) at each future time step based on the GARCH component

### Confidence Intervals

Forecast confidence intervals are computed assuming normally distributed forecast errors:
- **68% CI:** Approximately ±1 standard deviation from the mean
- **95% CI:** Approximately ±2 standard deviations from the mean

Note: As the forecast horizon increases, prediction intervals typically widen, reflecting increased uncertainty.

## Forecast Summary

| Statistic | Value |
|-----------|-------|
| Mean of Forecasts | {forecast_df['mean'].mean():.6f} |
| Std Dev of Forecasts | {forecast_df['mean'].std():.6f} |
| Min Forecast | {forecast_df['mean'].min():.6f} |
| Max Forecast | {forecast_df['mean'].max():.6f} |
| Average Forecast Std Dev | {forecast_df['std_dev'].mean():.6f} |

## Forecast Trajectory

{_get_image_markdown(plot_path, "Forecast Plot with Confidence Intervals", use_data_uri)}

The plot above shows the mean forecast (blue line) along with 68% and 95% confidence intervals.

## Detailed Forecast Table

| Step | Mean Forecast | Std Dev | 95% CI Lower | 95% CI Upper |
|------|---------------|---------|--------------|--------------|
"""
    
    # Add forecast table rows
    for _, row in forecast_df.iterrows():
        step = int(row['step'])
        mean = row['mean']
        std_dev = row['std_dev']
        ci_lower = mean - 1.96 * std_dev
        ci_upper = mean + 1.96 * std_dev
        report += f"| {step} | {mean:.6f} | {std_dev:.6f} | {ci_lower:.6f} | {ci_upper:.6f} |\n"
    
    report += f"""

## Key Insights

"""
    
    # Add insights based on forecast characteristics
    mean_forecast = forecast_df['mean'].values
    std_devs = forecast_df['std_dev'].values
    
    # Check forecast trend
    if len(mean_forecast) > 1:
        trend = mean_forecast[-1] - mean_forecast[0]
        if abs(trend) < 0.01 * abs(mean_forecast[0]):
            report += "- **Trend:** The forecast exhibits a relatively stable trajectory with minimal drift.\n"
        elif trend > 0:
            report += f"- **Trend:** The forecast shows an upward trend of approximately {trend:.4f} over the horizon.\n"
        else:
            report += f"- **Trend:** The forecast shows a downward trend of approximately {abs(trend):.4f} over the horizon.\n"
    
    # Check uncertainty evolution
    if len(std_devs) > 1:
        uncertainty_increase = std_devs[-1] / std_devs[0]
        if uncertainty_increase > 1.5:
            report += f"- **Uncertainty Growth:** Forecast uncertainty increases significantly (by {(uncertainty_increase-1)*100:.1f}%) over the horizon, indicating higher confidence in near-term predictions.\n"
        elif uncertainty_increase > 1.1:
            report += f"- **Uncertainty Growth:** Forecast uncertainty increases moderately (by {(uncertainty_increase-1)*100:.1f}%) over the horizon.\n"
        else:
            report += "- **Uncertainty:** Forecast uncertainty remains relatively stable across the horizon.\n"
    
    # Volatility assessment
    avg_vol = np.mean(std_devs)
    if avg_vol > 0:
        cv = np.std(mean_forecast) / abs(np.mean(mean_forecast)) if np.mean(mean_forecast) != 0 else float('inf')
        if not np.isinf(cv):
            report += f"- **Coefficient of Variation:** {cv:.4f} - "
            if cv < 0.5:
                report += "Relatively low variability in forecasts.\n"
            else:
                report += "Substantial variability in forecasts.\n"
    
    report += f"""

## Caveats and Considerations

1. **Forecast Horizon:** Forecast accuracy typically decreases as the horizon increases. Near-term forecasts (1-10 steps) are generally more reliable.

2. **Model Assumptions:** Forecasts assume:
   - Model structure remains appropriate for future observations
   - Parameters remain stable (no structural breaks)
   - No unforeseen shocks or regime changes

3. **Confidence Intervals:** 
   - Assume normally distributed forecast errors
   - Do not account for parameter estimation uncertainty
   - May understate true uncertainty in volatile markets

4. **Conditional Nature:** Forecasts are conditional on the model specification and historical data used for estimation.

5. **Use Case Dependent:** Forecasts should be interpreted in context:
   - Financial returns: Short horizons typically more useful
   - Volatility forecasts: May be more stable than mean forecasts

## Next Steps

1. **Validate Forecasts:** Compare with realized values when available to assess forecast accuracy

2. **Update Model:** Consider refitting the model periodically as new data becomes available:
   ```bash
   ag-viz fit -d updated_data.csv -a {model_json.get('spec', {}).get('arima', {}).get('p', 1)},{model_json.get('spec', {}).get('arima', {}).get('d', 0)},{model_json.get('spec', {}).get('arima', {}).get('q', 1)} -g {model_json.get('spec', {}).get('garch', {}).get('p', 1)},{model_json.get('spec', {}).get('garch', {}).get('q', 1)} -o updated_model.json
   ```

3. **Scenario Analysis:** Simulate multiple paths to understand the distribution of possible outcomes:
   ```bash
   ag-viz simulate -m model.json -p 1000 -n {horizon} -o scenarios.csv
   ```

4. **Combine with Domain Knowledge:** Integrate forecasts with expert judgment and market intelligence

## References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.

---

*Report generated by ag-viz on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path


def generate_diagnostics_report(
    model_json: Dict[str, Any],
    data: pd.DataFrame,
    diagnostics_json: Optional[Dict[str, Any]],
    plot_path: Path,
    output_path: Path,
    use_data_uri: bool = False
) -> Path:
    """
    Generate a Markdown report for diagnostic analysis results.
    
    Parameters
    ----------
    model_json : Dict[str, Any]
        Model specification and parameters from JSON.
    data : pd.DataFrame
        Original time series data.
    diagnostics_json : Optional[Dict[str, Any]]
        Diagnostic test results from JSON.
    plot_path : Path
        Path to the residual diagnostic plots image.
    output_path : Path
        Path where the Markdown report will be saved.
    use_data_uri : bool, optional
        If True, embed images as data URIs (default: False).
    
    Returns
    -------
    Path
        Path to the saved Markdown report.
    """
    model_spec = format_model_spec(model_json)
    n_obs = len(data)
    
    report = f"""# ARIMA-GARCH Diagnostic Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report presents comprehensive diagnostic analysis for a fitted **{model_spec}** model on **{n_obs} observations**.

## Model Specification

- **Model Type:** {model_spec}
- **Observations:** {n_obs}
- **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}

## Methodology

### Purpose of Diagnostic Analysis

Diagnostic tests assess whether the fitted model adequately captures the patterns in the data. Key aspects examined:

1. **Residual Independence:** Are residuals free from autocorrelation?
2. **Normality:** Do residuals follow a normal distribution?
3. **Heteroskedasticity:** Has the GARCH component adequately captured volatility clustering?
4. **Model Adequacy:** Does the model provide a good statistical fit?

### Diagnostic Tests

#### Ljung-Box Test
Tests for autocorrelation in residuals at multiple lags. 
- **Null Hypothesis:** Residuals are independently distributed (no autocorrelation)
- **Interpretation:** p-value > 0.05 suggests residuals are uncorrelated (desired)

#### Ljung-Box Test on Squared Residuals
Tests whether GARCH has captured all volatility clustering.
- **Null Hypothesis:** Squared residuals show no autocorrelation
- **Interpretation:** p-value > 0.05 suggests GARCH adequately models conditional variance

#### Jarque-Bera Test
Tests for normality of residuals.
- **Null Hypothesis:** Residuals are normally distributed
- **Interpretation:** p-value > 0.05 suggests approximate normality (though some deviation is common)

## Diagnostic Test Results

"""
    
    # Add test results if available
    if diagnostics_json:
        ljung_box = diagnostics_json.get('ljung_box_test', {})
        jarque_bera = diagnostics_json.get('jarque_bera_test', {})
        
        if ljung_box:
            report += "### Ljung-Box Test Results\n\n"
            report += "| Lag | Test Statistic | p-value | Result |\n"
            report += "|-----|----------------|---------|--------|\n"
            
            lags = ljung_box.get('lags', [])
            statistics = ljung_box.get('statistics', [])
            pvalues = ljung_box.get('pvalues', [])
            
            for i, (lag, stat, pval) in enumerate(zip(lags, statistics, pvalues)):
                result = "✓ Pass" if pval > 0.05 else "✗ Fail"
                report += f"| {lag} | {stat:.4f} | {pval:.4f} | {result} |\n"
            
            # Interpretation
            failing_tests = sum(1 for p in pvalues if p <= 0.05)
            if failing_tests == 0:
                report += "\n**Interpretation:** All Ljung-Box tests pass, indicating residuals are free from significant autocorrelation. The model adequately captures temporal dependencies.\n\n"
            elif failing_tests < len(pvalues) / 2:
                report += f"\n**Interpretation:** {failing_tests} out of {len(pvalues)} tests show some autocorrelation. Consider increasing model orders or investigating specific lags.\n\n"
            else:
                report += f"\n**Interpretation:** Significant autocorrelation detected in residuals. The model may be misspecified. Consider alternative model orders.\n\n"
        
        if jarque_bera:
            report += "### Jarque-Bera Normality Test\n\n"
            report += "| Statistic | Value |\n"
            report += "|-----------|-------|\n"
            report += f"| Test Statistic | {jarque_bera.get('statistic', 'N/A')} |\n"
            report += f"| p-value | {jarque_bera.get('pvalue', 'N/A')} |\n"
            
            pval = jarque_bera.get('pvalue', 1.0)
            if isinstance(pval, (int, float)):
                if pval > 0.05:
                    report += "\n**Interpretation:** Residuals appear approximately normally distributed (p > 0.05). This supports model assumptions.\n\n"
                else:
                    report += "\n**Interpretation:** Residuals deviate from normality (p ≤ 0.05). This is common in financial data and may suggest considering Student-t innovations or checking for outliers.\n\n"
    else:
        report += "*Diagnostic test results not available.*\n\n"
    
    report += f"""
## Residual Analysis Plots

{_get_image_markdown(plot_path, "Residual Diagnostic Plots", use_data_uri)}

The comprehensive diagnostic plot above includes:

1. **Standardized Residuals:** Should appear as white noise (random fluctuations around zero)
2. **Histogram:** Should approximate a normal distribution
3. **QQ-Plot:** Points should follow the diagonal line for normality
4. **ACF of Residuals:** Should show no significant autocorrelation (bars within confidence bands)
5. **ACF of Squared Residuals:** Should show no significant autocorrelation if GARCH adequately models volatility

## Key Findings

### Model Adequacy Assessment

"""
    
    # Provide assessment based on available information
    if diagnostics_json:
        ljung_box = diagnostics_json.get('ljung_box_test', {})
        pvalues = ljung_box.get('pvalues', [])
        
        if pvalues:
            passing_rate = sum(1 for p in pvalues if p > 0.05) / len(pvalues)
            
            if passing_rate > 0.8:
                report += "- **Overall Assessment:** The model demonstrates good fit with most diagnostic tests passing.\n"
            elif passing_rate > 0.5:
                report += "- **Overall Assessment:** The model shows acceptable fit, though some improvements may be possible.\n"
            else:
                report += "- **Overall Assessment:** The model may benefit from specification changes or alternative orders.\n"
        
        # Check normality
        jb_pval = diagnostics_json.get('jarque_bera_test', {}).get('pvalue', None)
        if jb_pval is not None and isinstance(jb_pval, (int, float)):
            if jb_pval < 0.01:
                report += "- **Normality:** Residuals show substantial departure from normality. Consider robust methods or alternative innovation distributions.\n"
            elif jb_pval < 0.05:
                report += "- **Normality:** Residuals show some departure from normality, which is common in practice.\n"
    else:
        report += "- Examine the residual plots above for visual assessment of model adequacy.\n"
    
    report += """

## Caveats and Considerations

1. **Diagnostic Limitations:**
   - Tests have varying power depending on sample size
   - Multiple testing increases chance of spurious rejections
   - Some tests (e.g., normality) are often violated in practice without severely impacting usefulness

2. **Practical vs. Statistical Significance:**
   - Slight deviations from ideal diagnostics may be acceptable
   - Consider both statistical tests and visual inspection
   - Economic significance may differ from statistical significance

3. **Model Refinement:**
   - Failed diagnostics suggest areas for improvement, not necessarily model failure
   - Consider both increasing and decreasing model complexity
   - Balance model complexity with interpretability and overfitting concerns

4. **Sample Size Effects:**
   - Diagnostic tests become more powerful with larger samples
   - May detect minor deviations that have little practical impact
   - With small samples, tests may lack power to detect real issues

## Next Steps

### If Diagnostics Are Satisfactory

1. **Proceed with Forecasting:**
   ```bash
   ag-viz forecast -m model.json -n 30 -o forecast.csv --markdown
   ```

2. **Generate Scenarios:**
   ```bash
   ag-viz simulate -m model.json -p 1000 -n 500 -o simulation.csv --markdown
   ```

### If Diagnostics Indicate Issues

1. **Try Alternative Specifications:**
   ```bash
   ag select -d data.csv -c BIC --max-p 3 --max-q 3 -o alternative_model.json
   ```

2. **Increase Model Orders:** If autocorrelation persists, try higher AR/MA orders

3. **Examine Outliers:** Investigate unusual observations that may affect fit

4. **Consider Extensions:**
   - Asymmetric GARCH models (if volatility responds differently to positive/negative shocks)
   - Student-t innovations (if heavy tails are present)
   - Seasonal components (if data exhibits seasonality)

## References

- Ljung, G. M., & Box, G. E. P. (1978). On a Measure of Lack of Fit in Time Series Models. Biometrika.
- Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence. Economics Letters.
- Engle, R. F., & Ng, V. K. (1993). Measuring and Testing the Impact of News on Volatility. Journal of Finance.

---

*Report generated by ag-viz on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path


def generate_simulation_report(
    model_json: Dict[str, Any],
    simulation_df: pd.DataFrame,
    plot_path: Path,
    output_path: Path,
    n_paths: int,
    length: int,
    use_data_uri: bool = False
) -> Path:
    """
    Generate a Markdown report for simulation results.
    
    Parameters
    ----------
    model_json : Dict[str, Any]
        Model specification and parameters from JSON.
    simulation_df : pd.DataFrame
        Simulation data with multiple paths.
    plot_path : Path
        Path to the simulation plot image.
    output_path : Path
        Path where the Markdown report will be saved.
    n_paths : int
        Number of simulation paths generated.
    length : int
        Length of each simulation path.
    use_data_uri : bool, optional
        If True, embed images as data URIs (default: False).
    
    Returns
    -------
    Path
        Path to the saved Markdown report.
    """
    model_spec = format_model_spec(model_json)
    
    report = f"""# ARIMA-GARCH Simulation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report presents results from simulating **{n_paths} paths** of length **{length}** from a fitted **{model_spec}** model.

## Model Specification

- **Model Type:** {model_spec}
- **Number of Paths:** {n_paths}
- **Path Length:** {length} observations
- **Date Generated:** {datetime.now().strftime('%Y-%m-%d')}

## Methodology

### Monte Carlo Simulation

Monte Carlo simulation generates multiple realizations (paths) from the fitted ARIMA-GARCH model to:

1. **Assess Uncertainty:** Understand the range of possible future outcomes
2. **Risk Analysis:** Quantify tail risks and extreme scenarios
3. **Scenario Planning:** Generate distributions for decision-making
4. **Model Validation:** Verify model behavior matches data characteristics

### Simulation Process

Each simulated path is generated by:
1. Drawing random innovations from the specified distribution (Normal or Student-t)
2. Applying ARIMA equations to generate returns
3. Applying GARCH equations to generate time-varying volatility
4. Maintaining consistency with the fitted model parameters

## Simulation Statistics

"""
    
    # Calculate statistics across all paths
    # simulation_df has columns: 'path', 'observation', 'return', 'volatility'
    all_values = simulation_df['return'].values
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
    
    if len(all_values) > 0:
        report += f"""
### Aggregate Statistics (All Paths)

| Statistic | Value |
|-----------|-------|
| Total Observations | {len(all_values)} |
| Mean | {np.mean(all_values):.6f} |
| Std Dev | {np.std(all_values):.6f} |
| Min | {np.min(all_values):.6f} |
| Max | {np.max(all_values):.6f} |
| Skewness | {stats.skew(all_values):.4f} |
| Kurtosis | {stats.kurtosis(all_values):.4f} |
| 5th Percentile | {np.percentile(all_values, 5):.6f} |
| 25th Percentile | {np.percentile(all_values, 25):.6f} |
| Median | {np.median(all_values):.6f} |
| 75th Percentile | {np.percentile(all_values, 75):.6f} |
| 95th Percentile | {np.percentile(all_values, 95):.6f} |

### Terminal Value Statistics (End of Horizon)

"""
        # Get terminal values (last observation of each path)
        # Group by path and get the last return value for each path
        terminal_values = simulation_df.groupby('path')['return'].last().values
        terminal_values = terminal_values[~np.isnan(terminal_values)]
        
        report += f"""
| Statistic | Value |
|-----------|-------|
| Mean Terminal Value | {np.mean(terminal_values):.6f} |
| Std Dev Terminal Value | {np.std(terminal_values):.6f} |
| Min Terminal Value | {np.min(terminal_values):.6f} |
| Max Terminal Value | {np.max(terminal_values):.6f} |
| 5th Percentile | {np.percentile(terminal_values, 5):.6f} |
| 95th Percentile | {np.percentile(terminal_values, 95):.6f} |

"""
    
    report += f"""
## Simulation Paths Visualization

{_get_image_markdown(plot_path, "Simulation Paths with Percentile Bands", use_data_uri)}

The plot above shows:
- **Individual Paths:** Sample trajectories from the simulation
- **Mean Path:** Average across all simulated paths
- **Percentile Bands:** Shaded regions showing 5th-95th and 25th-75th percentiles
- **Terminal Distribution:** Histogram of final values across all paths

## Key Insights

"""
    
    # Provide insights based on simulation characteristics
    if len(all_values) > 0:
        # Volatility assessment
        vol = np.std(all_values)
        report += f"- **Volatility:** The simulated paths exhibit a standard deviation of {vol:.4f}, "
        if vol > 0.1:
            report += "indicating substantial variability in potential outcomes.\n"
        elif vol > 0.05:
            report += "indicating moderate variability in potential outcomes.\n"
        else:
            report += "indicating relatively low variability in potential outcomes.\n"
        
        # Tail behavior
        skew = stats.skew(all_values)
        kurt = stats.kurtosis(all_values)
        
        if abs(skew) > 0.5:
            direction = "right" if skew > 0 else "left"
            report += f"- **Asymmetry:** Distribution is {direction}-skewed (skewness = {skew:.2f}), suggesting "
            if skew > 0:
                report += "more frequent large positive outcomes.\n"
            else:
                report += "more frequent large negative outcomes.\n"
        
        if kurt > 1.0:
            report += f"- **Tail Risk:** High kurtosis ({kurt:.2f}) indicates heavy tails with more extreme values than a normal distribution, suggesting non-negligible tail risk.\n"
        
        # Range of outcomes
        value_range = np.max(all_values) - np.min(all_values)
        report += f"- **Range of Outcomes:** Simulated values span a range of {value_range:.4f}, from {np.min(all_values):.4f} to {np.max(all_values):.4f}.\n"
        
        # Terminal value spread
        if len(terminal_values) > 0:
            terminal_range = np.percentile(terminal_values, 95) - np.percentile(terminal_values, 5)
            report += f"- **Terminal Uncertainty:** The 90% confidence interval for terminal values spans {terminal_range:.4f}, illustrating the degree of outcome uncertainty.\n"
    
    report += """

## Applications

### Risk Management

Use simulation results to:
- **Value at Risk (VaR):** Calculate percentiles for risk metrics
- **Stress Testing:** Assess model behavior under various scenarios
- **Tail Risk Analysis:** Examine extreme outcomes and their probabilities

### Decision Making

Simulations inform:
- **Capital Allocation:** Size positions based on potential outcomes
- **Hedging Strategies:** Design hedges that account for path dependency
- **Scenario Planning:** Prepare for range of possible futures

### Model Validation

Compare simulated characteristics with historical data:
- Do simulated volatilities match historical patterns?
- Are extreme events appropriately represented?
- Does the model capture key stylized facts of the data?

## Caveats and Considerations

1. **Model Dependence:**
   - Simulations are only as good as the underlying model
   - Model misspecification propagates to simulated paths
   - Historical parameter estimates may not apply to future

2. **Sampling Variability:**
   - Increasing the number of paths improves precision of percentile estimates
   - Consider running more paths for critical applications

3. **Path Independence:**
   - Each path is an independent realization
   - Real-world dynamics may involve feedback effects not captured by the model

4. **Innovation Distribution:**
   - Standard simulations use Normal innovations
   - Consider Student-t innovations if heavy tails are important
   - Extreme events may still be underestimated

5. **Stationarity Assumption:**
   - Simulations assume stable parameters throughout the horizon
   - Real markets may experience regime shifts or structural changes

## Next Steps

1. **Analyze Specific Scenarios:** Extract and study paths of particular interest

2. **Calculate Risk Metrics:** Use simulated distribution to compute:
   - Value at Risk (VaR) at various confidence levels
   - Expected Shortfall (Conditional VaR)
   - Maximum drawdown distributions

3. **Compare with Historical Data:** Validate that simulated characteristics match observed patterns

4. **Sensitivity Analysis:** Re-simulate with alternative model specifications to assess robustness:
   ```bash
   # Try alternative model orders
   ag-viz fit -d data.csv -a 2,0,2 -g 1,1 -o alt_model.json
   ag-viz simulate -m alt_model.json -p {n_paths} -n {length} -o alt_simulation.csv --markdown
   ```

5. **Extend Simulation:** For long-term planning, simulate longer horizons:
   ```bash
   ag-viz simulate -m model.json -p 1000 -n 5000 -o long_term_sim.csv --markdown
   ```

## References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.
- McNeil, A. J., Frey, R., & Embrechts, P. (2005). Quantitative Risk Management: Concepts, Techniques and Tools.

---

*Report generated by ag-viz on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return output_path
