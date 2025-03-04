# mass_functions:

import healpy as hp
import numpy as np
import pandas as pd

#from constants import *
from constants import MASS_BINS, DEX, NSIDE, REGIONS_OF_SKY
from richard_curve import get_distance_from_mass

"""
input:
dataframe of galaxies with information:
- mass
- completeness
- region

fucntion:   The radius is calculated for each galaxy based on its mass using Richard's curve, which provides a continuous radius value.
            These continuous radius values are used to compute the volume.
            Although the calculations are performed at discrete mass points (the masses of the galaxies in the list), the values themselves are continuous.
            The final volume calculation integrates these continuous values over the discrete set of masses.
            The completeness of each mass-bin is also calulated, so now we have the wight of mass bins based on the volume that re got from the rechard curve.

output:
galaxy mass function
"""


# Function to get the weighted histogram of the objects of each region
def get_weighted_mass_histogram(input_mass_completeness_dataframe: pd.DataFrame, region_name: str):
    """
    this is the main function
    """
    # filtered_by_region_dataframe = input_mass_completeness_dataframe[input_mass_completeness_dataframe['region'] == region_name]
    mass_column = input_mass_completeness_dataframe['mstar']
    completeness_column = input_mass_completeness_dataframe['completeness']
    # volume_richard_curve = get_region_volume(region_name=region_name, mass_list=mass_column)
    # weight = np.log(10) / (volume_richard_curve * completeness_column * DEX)
    weight = np.log(10) / (completeness_column * DEX)
    mass_histogram = np.histogram(mass_column, MASS_BINS, weights=weight)[0]
    histogram_errors = np.sqrt(np.histogram(mass_column, MASS_BINS, weights=weight**2)[0])

    return mass_histogram, histogram_errors



# Function to get the volume of each region
def get_region_volume(region_name: str, mass_list: list, mass_luminosity_cutoff):
    region = REGIONS_OF_SKY[region_name]
    region_area = calculate_patch_area(patch=region, nside=NSIDE)
    average_pixel_area = 4 * np.pi / (12 * NSIDE**2)
    total_area_sphere = hp.nside2npix(NSIDE) * average_pixel_area
    fraction_region = region_area / total_area_sphere
    # return calculate_volume([get_distance_from_mass(np.log10(mass)) for mass in mass_list], fraction_region)
    return calculate_volume([get_distance_from_mass(mass, mass_luminosity_cutoff) for mass in mass_list], fraction_region)



def get_cluster_volume(radius):
    return 4 / 3 * np.pi * radius ** 3


# Function to calculate the area for a given patch
def calculate_patch_area(patch: dict, nside: int=NSIDE):
    RA_min, RA_max = np.deg2rad(patch['RAcen'])
    DEC_min, DEC_max = np.deg2rad(patch['DECcen'])

    # Calculate the pixel indices for the given patch
    pix_indices = np.arange(hp.nside2npix(nside))
    pix_indices_patch = pix_indices[
        (hp.pixelfunc.pix2ang(nside, pix_indices)[0] >= np.pi/2 - DEC_max) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[0] <= np.pi/2 - DEC_min) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[1] >= RA_min) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[1] <= RA_max)
    ]

    # Calculate the area of the given patch using the average solid angle of a pixel
    average_pixel_area = 4 * np.pi / (12 * nside**2)
    patch_area = len(pix_indices_patch) * average_pixel_area

    return patch_area


# Function to calculate volume (Mpc^3)
def calculate_volume(radius, fraction):
    return 4 / 3 * np.pi * np.power(radius, 3) * fraction


def calculate_error(y_data, number_of_galaxies):
    return np.sqrt((y_data * number_of_galaxies**(-1/2))**2 + (y_data * 0.043)**2)
