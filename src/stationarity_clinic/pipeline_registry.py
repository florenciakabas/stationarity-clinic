"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from stationarity_clinic.pipelines import assessment_and_recommendation
from stationarity_clinic.pipelines import data_processing
from stationarity_clinic.pipelines import data_science


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    assessment_pipeline = assessment_and_recommendation.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()

    return {
        "assessment": assessment_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "__default__": assessment_pipeline + data_processing_pipeline + data_science_pipeline,
    }