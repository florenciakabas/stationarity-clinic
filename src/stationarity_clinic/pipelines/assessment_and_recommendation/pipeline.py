"""Pipeline for stationarity assessment and recommendation."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import assess_stationarity, extract_recommendations


def create_pipeline(**kwargs) -> Pipeline:
    """Create the stationarity assessment and recommendation pipeline.
    
    Returns:
        A pipeline that performs stationarity assessment and generates recommendations.
    """
    return pipeline(
        [
            node(
                func=assess_stationarity,
                inputs=["time_series", "params:stationarity_assessment"],
                outputs="stationarity_assessment_results",
                name="assess_stationarity_node",
            ),
            node(
                func=extract_recommendations,
                inputs="stationarity_assessment_results",
                outputs="stationarity_recommendations",
                name="extract_recommendations_node",
            ),
        ]
    )