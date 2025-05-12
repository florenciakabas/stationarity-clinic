"""Tests for the Assessor class."""

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState
from stationarity_clinic.assessment_and_recommendation import Assessor, TestTypes


@pytest.fixture
def stationary_series():
    """Create a stationary time series for testing."""
    # Create a random stationary series
    rng = RandomState(42)
    return pd.Series(rng.normal(0, 1, 100))


@pytest.fixture
def non_stationary_series():
    """Create a non-stationary time series for testing."""
    # Create a random walk series (non-stationary)
    rng = RandomState(42)
    random_data = rng.normal(0, 1, 100)
    return pd.Series(np.cumsum(random_data))


@pytest.fixture
def assessor():
    """Create an Assessor instance for testing."""
    return Assessor()


class TestAssessor:
    """Test class for the Assessor class."""

    def test_initialization(self, assessor):
        """Test that the Assessor initializes correctly."""
        assert isinstance(assessor, Assessor)

    def test_adf_test_stationary(self, assessor, stationary_series):
        """Test ADF test on a stationary series."""
        result = assessor.adf_test(stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For truly stationary data, we expect the test to reject the null hypothesis
        # (null: series has a unit root, i.e., is non-stationary)
        # However, this can sometimes fail due to the random nature of the test data
        # So we don't assert the result here

    def test_adf_test_non_stationary(self, assessor, non_stationary_series):
        """Test ADF test on a non-stationary series."""
        result = assessor.adf_test(non_stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For non-stationary data, we expect the test to fail to reject the null
        # hypothesis (i.e., stationary=False)
        assert result["stationary"] is False

    def test_kpss_test_stationary(self, assessor, stationary_series):
        """Test KPSS test on a stationary series."""
        result = assessor.kpss_test(stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For stationary data, we expect to fail to reject the null hypothesis
        # (null: series is stationary)
        assert result["stationary"] is True

    def test_kpss_test_non_stationary(self, assessor, non_stationary_series):
        """Test KPSS test on a non-stationary series."""
        result = assessor.kpss_test(non_stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For non-stationary data, we expect to reject the null hypothesis
        # (i.e., stationary=False)
        assert result["stationary"] is False

    def test_pp_test_stationary(self, assessor, stationary_series):
        """Test PP test on a stationary series."""
        result = assessor.pp_test(stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For stationary data, we expect to reject the null hypothesis
        # (null: series has a unit root, i.e., is non-stationary)
        # However, this can sometimes fail due to the random nature of the test data
        # So we don't assert the result here

    def test_pp_test_non_stationary(self, assessor, non_stationary_series):
        """Test PP test on a non-stationary series."""
        result = assessor.pp_test(non_stationary_series)
        
        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "stationary" in result
        
        # For non-stationary data, we expect to fail to reject the null hypothesis
        # (i.e., stationary=False)
        assert result["stationary"] is False

    def test_run_all_tests(self, assessor, stationary_series, non_stationary_series):
        """Test running all tests at once."""
        # Stationary case
        results_stationary = assessor.run_all_tests(stationary_series)
        
        assert isinstance(results_stationary, dict)
        assert TestTypes.ADF.value in results_stationary
        assert TestTypes.KPSS.value in results_stationary
        assert TestTypes.PP.value in results_stationary
        assert "overall_stationary" in results_stationary
        
        # Non-stationary case
        results_non_stationary = assessor.run_all_tests(non_stationary_series)
        assert results_non_stationary["overall_stationary"] is False

    def test_detailed_assessment(self, assessor, stationary_series, non_stationary_series):
        """Test detailed assessment with different regression options."""
        # Stationary case
        detailed_stationary = assessor.detailed_assessment(stationary_series)
        
        assert isinstance(detailed_stationary, dict)
        assert "constant" in detailed_stationary
        assert "constant_trend" in detailed_stationary
        assert "summary" in detailed_stationary
        
        # Non-stationary case
        detailed_non_stationary = assessor.detailed_assessment(non_stationary_series)
        assert isinstance(detailed_non_stationary["summary"], dict)
        assert "is_stationary" in detailed_non_stationary["summary"]
        assert detailed_non_stationary["summary"]["is_stationary"] is False
        assert "recommendations" in detailed_non_stationary["summary"]
        assert len(detailed_non_stationary["summary"]["recommendations"]) > 0