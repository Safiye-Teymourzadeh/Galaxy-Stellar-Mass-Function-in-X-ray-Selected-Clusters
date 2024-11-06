# completeness file:

import numpy as np
import pandas as pd
from constants import *

'''
input:
- RA
- Dec
- flux_type
- uberID

functions:  check the location and find the region.
            read the flux data, and keep them in different types.
            make some flux_intervals (for each type of flux) and get number of objects in each interval. It does this for the big and the small survey, separately.
            divide the value of the last step of the small survey, over the big survey. This give us the completness of each flux bin of each flux_type.
            add the "completeness" column to each uberID.

output:
- completeness of each flux_interval (that we can turn it into masses)


'''



FLUX_BINS = np.logspace(-5,-2,1000)


def create_completeness_dataframe(big_survey, small_survey, flux_type, region):
    object_completeness = calculate_completeness_of_objects(big_survey=big_survey, small_survey=small_survey, flux_type=flux_type, region=region)
    completeness_df = pd.DataFrame()
    objects_in_one_region = filter_objects_by_region(small_survey, REGIONS_OF_SKY[region])
    completeness_df['uberID'] = objects_in_one_region['uberID'].byteswap().newbyteorder()
    completeness_df['completeness'] = object_completeness
    completeness_df = completeness_df.dropna(subset=['completeness'])
    return completeness_df

def calculate_completeness_of_objects(big_survey, small_survey, flux_type, region):
    cumulative_completeness = get_cumulative_completeness(
        big_survey=big_survey,
        small_survey=small_survey,
        flux_type=flux_type,
        region=region
    )
    objects_in_one_region = filter_objects_by_region(small_survey, REGIONS_OF_SKY[region])
    return np.interp(objects_in_one_region[flux_type], FLUX_BINS[:-1], cumulative_completeness)

def get_cumulative_completeness(big_survey, small_survey, flux_type, region):
    number_of_obj_big_survey, number_of_obj_small_survey = get_stat(big_survey=big_survey, small_survey=small_survey, flux_type=flux_type,region=region)
    return np.cumsum(number_of_obj_small_survey)/np.cumsum(number_of_obj_big_survey)

def get_stat(big_survey, small_survey, flux_type, region):
    big_survey_region = filter_objects_by_region(big_survey, REGIONS_OF_SKY[region])
    small_survey_region = filter_objects_by_region(small_survey, REGIONS_OF_SKY[region])
    counts_of_big_survey = get_interval_counts_from_surveys(survey_name=big_survey_region , flux_type=flux_type)
    counts_of_small_survey = get_interval_counts_from_surveys(survey_name=small_survey_region , flux_type=flux_type)
    return counts_of_big_survey, counts_of_small_survey

def get_interval_counts_from_surveys(survey_name, flux_type):
    flux_list = read_flux(survey_name,flux_type)
    interval_counts = count_objects_in_flux_intervals(flux_list, FLUX_BINS)
    return interval_counts

def read_flux(survey_name, flux_type):
    return survey_name[flux_type]

def count_objects_in_flux_intervals(flux_data, intervals):
    counts = np.histogram(flux_data, bins=intervals)[0]
    return counts

def filter_objects_by_region(survey_name, region):
    RAcen_mask = (survey_name['RAcen'] >= region['RAcen'][0]) & (survey_name['RAcen'] <= region['RAcen'][1])
    DECcen_mask = (survey_name['DECcen'] >= region['DECcen'][0]) & (survey_name['DECcen'] <= region['DECcen'][1])
    return survey_name[RAcen_mask & DECcen_mask]


