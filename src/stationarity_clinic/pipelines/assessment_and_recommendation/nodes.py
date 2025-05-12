"""Pipeline nodes for stationarity assessment and recommendation."""

import logging
from typing import Dict, List, Union

import pandas as pd
from stationarity_clinic.assessment_and_recommendation import Assessor


def assess_stationarity(time_series: pd.Series, parameters: Dict) -> Dict:
    """Assess stationarity of a time series using statistical tests.
    
    Args:
        time_series: The time series data to assess.
        parameters: Parameters for the stationarity tests including:
            - alpha: Significance level for hypothesis testing
            - detailed: Whether to perform detailed assessment with multiple test configurations
            
    Returns:
        Dict containing assessment results from various stationarity tests.
    """
    alpha = parameters.get("alpha", 0.05)
    detailed = parameters.get("detailed", False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Assessing stationarity with alpha={alpha}, detailed={detailed}")
    
    assessor = Assessor()
    
    if detailed:
        results = assessor.detailed_assessment(time_series, alpha=alpha)
    else:
        results = assessor.run_all_tests(time_series, alpha=alpha)
    
    logger.info(f"Assessment complete. Overall stationarity: {results.get('overall_stationary', results.get('summary', {}).get('is_stationary', False))}")
    
    return results


def extract_recommendations(assessment_results: Dict) -> List[str]:
    """Extract recommendations from assessment results.
    
    Args:
        assessment_results: Results from stationarity assessment.
        
    Returns:
        List of recommendations for achieving stationarity.
    """
    logger = logging.getLogger(__name__)
    
    # Handle both detailed and simple assessment results
    if "summary" in assessment_results:
        # Detailed assessment
        recommendations = assessment_results["summary"].get("recommendations", [])
    else:
        # Simple assessment
        if not assessment_results.get("overall_stationary", True):
            recommendations = [
                "Consider differencing the time series.",
                "Log transformation may help stabilize variance.",
                "Seasonal adjustment might be necessary if seasonal patterns are present.",
                "Check for structural breaks or regime changes in the data."
            ]
        else:
            recommendations = ["Time series appears to be stationary."]
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    
    return recommendations