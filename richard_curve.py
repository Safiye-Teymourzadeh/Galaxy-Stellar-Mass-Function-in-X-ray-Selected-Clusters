import numpy as np
from numpy import sqrt, pi, histogram, linspace, cumsum
import pandas as pd
from astropy import units as u
from constants import SUN_ABSOLUTE_MAGNITUDE, LOG_MASS_LUMINOSITY_RATIO_BINS


def filter_for_richards_curve(richards_curve:np.array, masses_of_richards_curve:np.array, mass_of_galaxy:float, distance_of_galaxy:float) -> bool:
    max_distance = np.interp(mass_of_galaxy, masses_of_richards_curve, richards_curve)
    if distance_of_galaxy > max_distance:
        return False
    else:
        return True

def get_distance_from_mass(mass_in_msol: float, log_cutoff_mass_to_light_ratio: float) -> float:
    # uses magnitude limit in r_band of 19.65 as mentioned in dDriver
    return 10**(19.65 / 5 + 1 + 0.5 * (np.log10(mass_in_msol) - log_cutoff_mass_to_light_ratio) - SUN_ABSOLUTE_MAGNITUDE / 5) / 10 ** 6


def get_mass_luminosity_cutoff(galaxy_df: pd.DataFrame, cut_off_percentage: float) -> float:
    '''
    calculate the M/L cutoff value for a survey of galaxies. the dataframe needs to have the following columns:
    comoving_distance (directly from the data columns)
    mstar
    flux_rt
    '''
    # get the mass luminosity histogram of our survey
    log_mass_luminosity_ratio_histogram = get_mass_luminosity_histogram(galaxy_df)[0]

    # normalize the histogram so it becomes a histogram of percentages
    histogram_of_percentages = log_mass_luminosity_ratio_histogram / len(galaxy_df) * 100
    # get the cumsum of the histogram
    cumsum_of_histogram = cumsum(histogram_of_percentages)
    # the ID of the bin is now that where the cumsum becomes larger than 95(%)
    bin_id = len(cumsum_of_histogram[cumsum_of_histogram < cut_off_percentage])
    # now we can find the center of the bin with ID bin_id which gives us our mass-luminosity-ratio cutoff
    return LOG_MASS_LUMINOSITY_RATIO_BINS[bin_id]


def get_mass_luminosity_histogram(galaxy_df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # get the luminosity distance from z
    # distances = cosmo.comoving_distance(galaxy_df['z']).value
    distances = galaxy_df['comoving_distance']
    # get the luminosity from flux and distance
    luminosities = [get_luminosity_from_flux_and_distance(flux, distance) for (flux, distance) in
                    zip(galaxy_df['flux_rt'], distances)]
    # calculate the ratios
    log_mass_luminosity_ratios = np.log10(galaxy_df['mstar'] / luminosities)
    # create the ratio-histogram
    return histogram(log_mass_luminosity_ratios, LOG_MASS_LUMINOSITY_RATIO_BINS)


def get_luminosity_from_flux_and_distance(flux: float, distance: float) -> float:
    # takes flux in Jansky and distance in Mpc
    # converts the units
    # returns luminosity in solar luminosities
    flux_in_jansky = flux * u.Jy
    distance_in_mpc = distance * u.Mpc
    effective_frequency = 4.87 * 10 ** 14 / u.s

    luminosity = 4 * pi * distance_in_mpc ** 2 * flux_in_jansky * effective_frequency
    solar_luminosity = 1 * u.solLum
    return luminosity.decompose() / solar_luminosity.decompose()