import pandas as pd
import sys
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from constants import c



"""
what is this function for?

input:
dataframe of galaxies
- galaxy ID
- redshift
- dec/ ra
dataframe of clusters
- cluster ID
- redshift
- dec/ ra
- mass
- velocity dispersion
- r_lambda Radius

output:
dataframe of galaxies
- galaxy ID
- redshift
- dec/ ra
- Field or Cluster Member
- mass of Cluster if member
- cluster ID if member
"""


# def match_galaxies_and_clusters(galaxy_dataframe: pd.DataFrame, cluster_dataframe: pd.DataFrame):
#     """
#     This is the main function.
#     """
#     galaxy_dataframe['environment'] = "Field"
#     galaxy_dataframe['cluster_mass'] = None
#     galaxy_dataframe['cluster_name'] = None
#
#     for i, galaxy in galaxy_dataframe.iterrows():
#
#         for j, cluster in cluster_dataframe.iterrows():
#             if compare_position(galaxy_ra=galaxy['RA'], galaxy_dec=galaxy['DEC'], cluster_ra=cluster['RA'], cluster_dec=cluster['DEC'], cluster_z=cluster['z'], cluster_radius_Mpc=cluster['cluster_radius_Mpc']):
#                 if compare_redshift(cluster_z=cluster['z'], cluster_Velocity_Dispersion=cluster['cluster_Velocity_Dispersion'], galaxy_z=galaxy['z']):
#                     galaxy_dataframe.at[i, 'environment'] = "ClusterMember"
#                     galaxy_dataframe.at[i, 'cluster_radius'] = cluster['cluster_radius_Mpc']
#                     # galaxy_dataframe.at[i, 'cluster_MASS'] = cluster['cluster_MASS']
#                     galaxy_dataframe.at[i, 'cluster_name'] = cluster['c_NAME']
#
#         sys.stdout.write('\r')
#         sys.stdout.write(f'Progress: {i / len(galaxy_dataframe) * 100}%')
#         sys.stdout.flush()
#     return galaxy_dataframe


def match_galaxies_and_clusters(galaxy_dataframe: pd.DataFrame, cluster_dataframe: pd.DataFrame):
    """
    This is the main function.
    """
    galaxy_dataframe['environment'] = "Field"
    galaxy_dataframe['cluster_mass'] = None
    galaxy_dataframe['cluster_name'] = None

    for i, galaxy in galaxy_dataframe.iterrows():
        for j, cluster in cluster_dataframe.iterrows():
            if compare_position(
                    galaxy_ra=galaxy['RA'], galaxy_dec=galaxy['DEC'],
                    cluster_ra=cluster['RA'], cluster_dec=cluster['DEC'],
                    cluster_z=cluster['z'], cluster_radius_Mpc=cluster['cluster_radius_Mpc']
            ):

                # Check if the cluster has velocity dispersion
                if not pd.isna(cluster['cluster_Velocity_Dispersion']):
                    redshift_match = compare_redshift(
                        cluster_z=cluster['z'],
                        cluster_Velocity_Dispersion=cluster['cluster_Velocity_Dispersion'],
                        galaxy_z=galaxy['z']
                    )
                else:
                    # If no velocity dispersion, check BEST_Z_TYPE_1
                    if cluster['z_type'] != "photo_z":
                        redshift_match = compare_redshift(
                            cluster_z=cluster['z'],
                            cluster_Velocity_Dispersion=0,  # Effectively ignores velocity dispersion
                            galaxy_z=galaxy['z']
                        )
                    else:
                        redshift_match = False

                if redshift_match:
                    galaxy_dataframe.at[i, 'environment'] = "ClusterMember"
                    galaxy_dataframe.at[i, 'cluster_radius'] = cluster['cluster_radius_Mpc']
                    galaxy_dataframe.at[i, 'cluster_name'] = cluster['c_NAME']

        sys.stdout.write('\r')
        sys.stdout.write(f'Progress: {i / len(galaxy_dataframe) * 100}%')
        sys.stdout.flush()
    return galaxy_dataframe


def compare_redshift(cluster_z, cluster_Velocity_Dispersion, galaxy_z, threshold_factor=1):
    delta_z = (cluster_Velocity_Dispersion / c) * threshold_factor * (1 + cluster_z) # added *(1+cluster_z) because of relativity redshift
    return np.abs(galaxy_z - cluster_z) <= 3 * delta_z  # should have three sigmas for the velocity dispersion


def angular_separation(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    delta_RA = ra2 - ra1
    delta_DEC = dec2 - dec1
    w = np.sin(delta_DEC/2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(delta_RA/2.0)**2
    return 2 * np.arcsin(np.sqrt(w))


def projected_distance(cluster_ra, cluster_dec, cluster_z, galaxy_ra, galaxy_dec):
    angular_dist = angular_separation(cluster_ra, cluster_dec, galaxy_ra, galaxy_dec)
    D_A = cosmo.angular_diameter_distance(cluster_z).value
    projected_distance_Mpc = D_A * angular_dist
    return projected_distance_Mpc


def compare_position(galaxy_ra, galaxy_dec, cluster_ra, cluster_dec, cluster_z, cluster_radius_Mpc) -> bool:
    projected_dist_Mpc = projected_distance(cluster_ra, cluster_dec, cluster_z, galaxy_ra, galaxy_dec)
    return projected_dist_Mpc <= cluster_radius_Mpc

