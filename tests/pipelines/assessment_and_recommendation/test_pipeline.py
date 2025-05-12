"""Tests for the stationarity assessment and recommendation pipeline."""

import logging
import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from stationarity_clinic.pipelines.assessment_and_recommendation import create_pipeline
from stationarity_clinic.pipelines.assessment_and_recommendation.nodes import assess_stationarity, extract_recommendations


@pytest.fixture
def stationary_series():
    """Create a stationary time series for testing."""
    # Create a random stationary series
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, 100), name="stationary_data")


@pytest.fixture
def non_stationary_series():
    """Create a non-stationary time series for testing."""
    # Create a random walk series (non-stationary)
    np.random.seed(42)
    random_data = np.random.normal(0, 1, 100)
    return pd.Series(np.cumsum(random_data), name="non_stationary_data")


@pytest.fixture
def assessment_parameters():
    """Create parameters for stationarity assessment."""
    return {
        "alpha": 0.05,
        "detailed": False
    }


def test_assess_stationarity(non_stationary_series, assessment_parameters):
    """Test the assess_stationarity node with non-stationary data."""
    results = assess_stationarity(non_stationary_series, assessment_parameters)
    
    assert isinstance(results, dict)
    assert "adf" in results
    assert "kpss" in results
    assert "pp" in results
    assert "overall_stationary" in results
    assert results["overall_stationary"] is False


def test_extract_recommendations():
    """Test the extract_recommendations node."""
    # Simple assessment results
    assessment_results = {
        "adf": {"stationary": False},
        "kpss": {"stationary": False},
        "pp": {"stationary": False},
        "overall_stationary": False
    }
    
    recommendations = extract_recommendations(assessment_results)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert "differencing" in recommendations[0]


def test_assessment_pipeline(caplog, non_stationary_series, assessment_parameters):
    """Test the full assessment pipeline."""
    pipeline = create_pipeline()
    catalog = DataCatalog()
    catalog.add_feed_dict({
        "time_series": non_stationary_series,
        "params:stationarity_assessment": assessment_parameters
    })
    
    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."
    
    result = SequentialRunner().run(pipeline, catalog)
    
    assert successful_run_msg in caplog.text
    assert "stationarity_assessment_results" in result
    assert "stationarity_recommendations" in result