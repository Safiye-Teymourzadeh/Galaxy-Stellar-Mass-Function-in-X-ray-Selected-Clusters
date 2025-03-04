import pandas as pd
import numpy as np

from completeness import FLUX_BINS

def calculate_completeness_of_objects(big_survey: pd.DataFrame, small_survey: pd.DataFrame) -> pd.DataFrame:
    # calculates the cumulative completeness of each object in the small_survey
    # it does that by comparing the number of objects of big vs small survey in each flux bin
    # use this function:
    # input: big_survey:   dataframe with a column called "flux" that has all objects of a survey
    #        small_survey: dataframe with a column called "flux" that only has the objects with known redshift
    # output: the small_survey dataframe with an added column for "completeness"
    cumulative_completeness = get_cumulative_completeness(
        big_survey=big_survey,
        small_survey=small_survey
    )
    object_completeness = np.interp(small_survey["flux"], FLUX_BINS[:-1], cumulative_completeness)
    small_survey["completeness"] = object_completeness
    return small_survey

def get_cumulative_completeness(big_survey: pd.DataFrame, small_survey: pd.DataFrame):
    number_of_obj_big_survey, number_of_obj_small_survey = get_stat(big_survey=big_survey, small_survey=small_survey)
    return np.cumsum(number_of_obj_small_survey)/np.cumsum(number_of_obj_big_survey)


def get_stat(big_survey: pd.DataFrame, small_survey: pd.DataFrame):
    counts_of_big_survey = get_interval_counts_from_surveys(survey=big_survey)
    counts_of_small_survey = get_interval_counts_from_surveys(survey=small_survey)
    return counts_of_big_survey, counts_of_small_survey

def get_interval_counts_from_surveys(survey):
    counts = np.histogram(survey["flux"], bins=FLUX_BINS)[0]
    return counts