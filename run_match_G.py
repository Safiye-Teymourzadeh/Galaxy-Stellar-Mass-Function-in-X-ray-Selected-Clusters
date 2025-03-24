from os import environ
import astropy.io.fits as fits
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo

from galaxy_cluster_matching import match_galaxies_and_clusters
from completeness_new import calculate_completeness_of_objects
from mass_function import get_weighted_mass_histogram, get_cluster_volume
from selection_function import get_mass_luminosity_cutoff, get_distance_from_mass, filter_for_selection_function, get_mass_luminosity_histogram
from constants import MASS_BINS, Z_MAX, LOG_MASS_LUMINOSITY_RATIO_BINS

# Computer
GAMA_Full_raw = fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits')[1].data
GAMA_raw = fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/merged/gkvscienceStellarmass_Morphology/fullmerged_gkvscienceStellarmass_Morphologyv02_shrinked.fits')[1].data #galaxy
eRASS1 = fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/eRASS/merged_primary&optical_clusters/merged_optical_primary_clusters.fits')[1].data #cluster

# Make dataframes
# the survey that has full observation with all different types of objects
gama_full_raw_df_unmasked = pd.DataFrame({
        'uberID': GAMA_Full_raw['uberID'].byteswap().newbyteorder(),
        'uberclass': GAMA_Full_raw['uberclass'].byteswap().newbyteorder(),
        'RA': GAMA_Full_raw['RAcen'].byteswap().newbyteorder(),
        'DEC': GAMA_Full_raw['Deccen'].byteswap().newbyteorder(),
        'z': GAMA_Full_raw['Z'].byteswap().newbyteorder(),
        'duplicate': GAMA_Full_raw['duplicate'].byteswap().newbyteorder(),
        'mask': GAMA_Full_raw['mask'].byteswap().newbyteorder(),
        'starmask': GAMA_Full_raw['starmask'].byteswap().newbyteorder(),
        'flux': GAMA_Full_raw['flux_rt'].byteswap().newbyteorder(), # Jansky
})
# remove all the objects that do not have flux
gama_full_raw_df_unmasked = gama_full_raw_df_unmasked[gama_full_raw_df_unmasked['flux'].notna()]

# the small survey that has all the parameters that we need
gama_df_unmasked = pd.DataFrame({
    'uberID': GAMA_raw['uberID'].byteswap().newbyteorder(),
    'RA': GAMA_raw['RAcen'].byteswap().newbyteorder(),
    'DEC': GAMA_raw['Deccen'].byteswap().newbyteorder(),
    'z': GAMA_raw['Z'].byteswap().newbyteorder(),
    'mstar': GAMA_raw['mstar'].byteswap().newbyteorder(),
    'flux': GAMA_raw['flux_rt'].byteswap().newbyteorder(), # Jansky
    'log_age': GAMA_raw['logage'].byteswap().newbyteorder(),
    'log_met': GAMA_raw['logtau'].byteswap().newbyteorder(),
    'log_met': GAMA_raw['logmet'].byteswap().newbyteorder(),
    'morphology': GAMA_raw['FINAL_CLASS'].byteswap().newbyteorder(),
    'uberclass': GAMA_raw['uberclass'].byteswap().newbyteorder(),  # Add missing columns
    'duplicate': GAMA_raw['duplicate'].byteswap().newbyteorder(),
    'mask': GAMA_raw['mask'].byteswap().newbyteorder(),
    'starmask': GAMA_raw['starmask'].byteswap().newbyteorder(),
    'NQ': GAMA_raw['NQ'].byteswap().newbyteorder(),
})

# Remove rows where flux is NaN
gama_df_unmasked = gama_df_unmasked[gama_df_unmasked['flux'].notna()]   # Jansky
gama_df_unmasked['comoving_distance'] = cosmo.comoving_distance(gama_df_unmasked['z']).value    # Mpc


# load the clusters of galaxies data from the fits object into dataframes
cluster_df_raw = pd.DataFrame({
    'c_ID': eRASS1['DETUID'].byteswap().newbyteorder(),
    'c_NAME': eRASS1['NAME'].byteswap().newbyteorder(),
    'RA': eRASS1['RA'].byteswap().newbyteorder(),
    'DEC': eRASS1['DEC'].byteswap().newbyteorder(),
    'z': eRASS1['BEST_Z'].byteswap().newbyteorder(),
    'z_type': eRASS1['BEST_Z_TYPE_1'].byteswap().newbyteorder(),
    'cluster_radius_kpc': eRASS1['R500'].byteswap().newbyteorder(),
    'cluster_Velocity_Dispersion': eRASS1['VDISP_BOOT'].byteswap().newbyteorder(),
})
cluster_df_raw['cluster_radius_Mpc'] = cluster_df_raw['cluster_radius_kpc'] / 1000
cluster_df_raw['distance'] = cosmo.comoving_distance(cluster_df_raw['z']).value     # Mpc
cluster_df_raw['cluster_volume'] = get_cluster_volume(cluster_df_raw['cluster_radius_Mpc'])


# Masks
# GALAXIES

# BIG GALAXY
# Apply the masks to the GAMA big DataFrame
gama_full_df = gama_full_raw_df_unmasked[
    (gama_full_raw_df_unmasked['uberclass'] == 1) &  # Classified as galaxy
    (gama_full_raw_df_unmasked['duplicate'] == 0) &  # Unique object
    (gama_full_raw_df_unmasked['mask'] == False) &   # Not masked
    (gama_full_raw_df_unmasked['starmask'] == False) &  # Not star-masked
    (gama_full_raw_df_unmasked['z'] < 0.4) &         # Redshift less than 0.4
    (gama_full_raw_df_unmasked['flux'] >= 5.011928e-05)  # Flux greater than or equal to 5.011928e-05
]


# SMALL GALAXY survey masks
gama_df = gama_df_unmasked[
    (gama_df_unmasked['uberclass'] == 1) &           # Classified as galaxy
    (gama_df_unmasked['duplicate'] == 0) &           # Unique object
    (gama_df_unmasked['mask'] == False) &            # Not masked
    (gama_df_unmasked['starmask'] == False) &        # Not star-masked
    (gama_df_unmasked['mstar'] > 0) &                # Stellar mass greater than 0
    (gama_df_unmasked['NQ'] > 2) &                   # Reliable redshift
    (gama_df_unmasked['z'] != 0) &                   # Redshift not zero
    (gama_df_unmasked['z'] != -9.999) &              # Redshift not -9.999 (invalid value)
    (gama_df_unmasked['z'] < Z_MAX) &                # Redshift less than Z_MAX (define Z_MAX)
    (gama_df_unmasked['flux'] >= 5.011928e-05)       # Flux greater than or equal to 5.011928e-05
]


# CLUSTERS
# Remove rows where 'cluster_Velocity_Dispersion' is NaN
cluster_df_noVelDisp = cluster_df_raw#.dropna(subset=['cluster_Velocity_Dispersion'])

cluster_df = cluster_df_noVelDisp[
    (cluster_df_noVelDisp['z'] <= 0.4) &
    (cluster_df_noVelDisp['cluster_radius_kpc'] != -1)
]


# Assuming you have all necessary imports and the constants file loaded correctly
gama_df = calculate_completeness_of_objects(gama_full_df, gama_df)


# LOOOOOOOOOOOOONG
# match the galaxies with the clusters
matched_gama_dataframe = match_galaxies_and_clusters(galaxy_dataframe=gama_df, cluster_dataframe=cluster_df)
matched_gama_dataframe.to_csv('gama_matched_in_erass1.csv')



if __name__ == "__main__":
    matched_gama_dataframe = match_galaxies_and_clusters(galaxy_dataframe=gama_df, cluster_dataframe=cluster_df)
    matched_gama_dataframe.to_csv('gama_matched_in_erass1.csv', index=False)
    print("Matching complete. Saved as gama_matched_in_erass1.csv")
