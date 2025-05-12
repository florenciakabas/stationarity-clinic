"""Stationarity Assessment module for time series analysis.

This module provides a class for formal statistical tests to assess stationarity
in time series data, including Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin
(KPSS), and Phillips-Perron (PP) tests.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


class TestTypes(Enum):
    """Enum for types of stationarity tests."""
    ADF = "adf"
    KPSS = "kpss"
    PP = "pp"


class Assessor:
    """Class for assessing stationarity in time series data.
    
    This class provides methods to perform formal statistical tests for stationarity,
    including Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS),
    and Phillips-Perron (PP) tests.
    """
    
    def __init__(self):
        """Initialize the Assessor class."""
        self.logger = logging.getLogger(__name__)
    
    def adf_test(self, 
                 time_series: Union[pd.Series, np.ndarray], 
                 regression: str = 'c', 
                 max_lags: Optional[int] = None,
                 alpha: float = 0.05) -> Dict:
        """Perform the Augmented Dickey-Fuller test for stationarity.
        
        The null hypothesis of the ADF test is that the time series has a unit root,
        meaning it is non-stationary. The alternative hypothesis is that the time series
        is stationary.
        
        Args:
            time_series: The time series data to test.
            regression: The type of regression to include in the test:
                        'c' - constant only (default)
                        'ct' - constant and trend
                        'ctt' - constant, linear and quadratic trend
                        'n' - no constant, no trend
            max_lags: Maximum number of lags to include. If None, it's calculated based on data.
            alpha: Significance level for hypothesis testing.
            
        Returns:
            Dict containing test results including:
                - test_statistic: The test statistic
                - p_value: The p-value of the test
                - critical_values: Critical values at different significance levels
                - stationary: Boolean indicating whether the series is stationary
        """
        result = adfuller(time_series, regression=regression, maxlag=max_lags)
        
        test_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        stationary = p_value < alpha
        
        self.logger.info(f"ADF Test - Test Statistic: {test_statistic:.4f}, p-value: {p_value:.4f}")
        self.logger.info(f"ADF Test - Stationary: {stationary}")
        
        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "stationary": stationary
        }
    
    def kpss_test(self, 
                  time_series: Union[pd.Series, np.ndarray], 
                  regression: str = 'c', 
                  nlags: Optional[int] = None,
                  alpha: float = 0.05) -> Dict:
        """Perform the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
        
        The null hypothesis of the KPSS test is that the time series is stationary.
        The alternative hypothesis is that the time series is non-stationary.
        
        Args:
            time_series: The time series data to test.
            regression: The type of regression to include in the test:
                        'c' - constant only (default)
                        'ct' - constant and trend
            nlags: Number of lags to use in the test. If None, it's calculated based on data.
            alpha: Significance level for hypothesis testing.
            
        Returns:
            Dict containing test results including:
                - test_statistic: The test statistic
                - p_value: The p-value of the test
                - critical_values: Critical values at different significance levels
                - stationary: Boolean indicating whether the series is stationary
        """
        result = kpss(time_series, regression=regression, nlags=nlags)
        
        test_statistic = result[0]
        p_value = result[1]
        critical_values = result[3]
        stationary = p_value > alpha  # Note: KPSS null hypothesis is that the series is stationary
        
        self.logger.info(f"KPSS Test - Test Statistic: {test_statistic:.4f}, p-value: {p_value:.4f}")
        self.logger.info(f"KPSS Test - Stationary: {stationary}")
        
        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "stationary": stationary
        }
    
    def pp_test(self, 
               time_series: Union[pd.Series, np.ndarray], 
               regression: str = 'c', 
               alpha: float = 0.05) -> Dict:
        """Perform the Phillips-Perron test for stationarity.
        
        The null hypothesis of the PP test is that the time series has a unit root,
        meaning it is non-stationary. The alternative hypothesis is that the time series
        is stationary.
        
        Args:
            time_series: The time series data to test.
            regression: The type of regression to include in the test:
                        'c' - constant only (default)
                        'ct' - constant and trend
                        'n' - no constant, no trend
            alpha: Significance level for hypothesis testing.
            
        Returns:
            Dict containing test results including:
                - test_statistic: The test statistic
                - p_value: The p-value of the test
                - critical_values: Critical values at different significance levels
                - stationary: Boolean indicating whether the series is stationary
        """
        # Phillips-Ouliaris is used here as it provides similar test functionality to PP
        # result = phillips_ouliaris(time_series, trend=regression)
        result = 1
        
        test_statistic = result.stat
        p_value = result.pvalue
        critical_values = {}  # Phillips-Ouliaris doesn't directly return critical values
        stationary = p_value < alpha
        
        self.logger.info(f"Phillips-Perron Test - Test Statistic: {test_statistic:.4f}, p-value: {p_value:.4f}")
        self.logger.info(f"Phillips-Perron Test - Stationary: {stationary}")
        
        return {
            "test_statistic": test_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "stationary": stationary
        }
    
    def run_all_tests(self, 
                     time_series: Union[pd.Series, np.ndarray],
                     alpha: float = 0.05,
                     regression: str = 'c') -> Dict[str, Dict]:
        """Run all stationarity tests on the time series.
        
        Args:
            time_series: The time series data to test.
            alpha: Significance level for hypothesis testing.
            regression: The type of regression to include in the tests (default: 'c').
                        Note that not all tests support all regression types.
                      
        Returns:
            Dict containing results of all tests with test names as keys.
        """
        results = {}
        
        # ADF Test
        results[TestTypes.ADF.value] = self.adf_test(time_series, regression=regression, alpha=alpha)
        
        # KPSS Test - Note that KPSS only supports 'c' and 'ct'
        kpss_regression = regression if regression in ['c', 'ct'] else 'c'
        results[TestTypes.KPSS.value] = self.kpss_test(time_series, regression=kpss_regression, alpha=alpha)
        
        # PP Test
        results[TestTypes.PP.value] = self.pp_test(time_series, regression=regression, alpha=alpha)
        
        # Overall assessment - Series is considered stationary if majority of tests indicate stationarity
        test_results = [test["stationary"] for test in results.values()]
        stationary_count = sum(test_results)
        results["overall_stationary"] = stationary_count >= len(test_results) / 2
        
        self.logger.info(f"Overall assessment - Stationary: {results['overall_stationary']}")
        
        return results
    
    def detailed_assessment(self, 
                           time_series: Union[pd.Series, np.ndarray],
                           alpha: float = 0.05) -> Dict:
        """Perform detailed stationarity assessment with multiple test configurations.
        
        This method runs stationarity tests with different regression options to provide
        a more comprehensive assessment.
        
        Args:
            time_series: The time series data to test.
            alpha: Significance level for hypothesis testing.
            
        Returns:
            Dict containing detailed test results.
        """
        regression_options = {
            "constant": "c",
            "constant_trend": "ct"
        }
        
        detailed_results = {}
        for name, regression in regression_options.items():
            detailed_results[name] = self.run_all_tests(time_series, alpha=alpha, regression=regression)
        
        # Generate summary recommendations
        summary = self._generate_summary(detailed_results)
        detailed_results["summary"] = summary
        
        return detailed_results
    
    def _generate_summary(self, detailed_results: Dict) -> Dict:
        """Generate a summary and recommendations based on detailed test results.
        
        Args:
            detailed_results: The detailed results from stationarity tests.
            
        Returns:
            Dict containing summary and recommendations.
        """
        # Count overall stationary results
        stationary_count = sum(config.get("overall_stationary", False) 
                              for config in detailed_results.values() 
                              if isinstance(config, dict) and "overall_stationary" in config)
        
        total_configs = sum(1 for config in detailed_results.values() 
                           if isinstance(config, dict) and "overall_stationary" in config)
        
        # Determine overall stationarity
        is_stationary = stationary_count >= total_configs / 2
        
        # Generate recommendations
        recommendations = []
        if not is_stationary:
            recommendations.append("Consider differencing the time series.")
            recommendations.append("Log transformation may help stabilize variance.")
            recommendations.append("Seasonal adjustment might be necessary if seasonal patterns are present.")
            recommendations.append("Check for structural breaks or regime changes in the data.")
        
        return {
            "is_stationary": is_stationary,
            "stationary_count": stationary_count,
            "total_configurations": total_configs,
            "recommendations": recommendations if not is_stationary else ["Time series appears to be stationary."]
        }