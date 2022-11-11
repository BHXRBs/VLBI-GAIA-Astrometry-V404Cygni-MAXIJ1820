"""
Skymotion package, developed to estimate projected motion of a celestial object,
considering parallax, proper motion, and in case of binaries, orbital motion.

This script is part of the modeling package to fit astrometric data.

Developed by Arash Bahramian and James Miller-Jones. 

Version: 201905A
"""

import numpy as np
from skyfield.api import Loader, load
#load = Loader('/home/arash/astro_sw/skyfield-data')

def earth_position(t):
    """
    Calculates Earth position (X,Y,Z) at time t using Python package Skyfield (http://rhodesmill.org/skyfield/).
    
    Parameters
    ----------
    t: Time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  

    Returns
    -------
    earth_pos: Earth positions (X,Y,Z)
               in the form an array with shape [3 * N] where N is length of t, values in unit of au.

    """
    planets = load('de438.bsp')
    earth = planets['earth']
    ts = load.timescale()
    times = ts.tdb_jd(t+2400000.5)
    earth_pos = earth.at(times).position.au
    return earth_pos


def t_0(t):
    """
    Calculates the midpoint time between first and last observations.
    Note that the value is rounded using "np.floor" for convenience.
    
    Parameters
    ----------
    t: Time
       an array (preferred to be in MJD format and Barycentric Dynamical Time (TDB) scale).
    
    Returns
    -------
    t_midpoint: A single value representing the midpoint time.
                
    
    """
    return np.floor(t.min()+((t.max() - t.min())/2.0))

def radians(angle):
    return angle*np.pi/180


def frac_parallax(t, alpha, delta):
    """
    Calculates fractions of parallax projected on RA and Dec axes.
    
    Parameters
    ----------
    t: Time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    
    alpha: Observation RA, in degrees
    
    delta: Observation Dec, in degrees
    
    Returns
    -------
    f_alpha, f_delta: Ra and Dec Parallax fractions respectively, at time t (unitless fractions). 

    
    """
    alpha_radians = radians(alpha)
    delta_radians = radians(delta)
    X, Y, Z = earth_position(t)
    f_alpha = ((X * np.sin(alpha_radians)) - (Y * np.cos(alpha_radians)))   
    f_delta = (X * np.cos(alpha_radians) * np.sin(delta_radians)) + (Y * np.sin(alpha_radians) * np.sin(delta_radians)) - (Z * np.cos(delta_radians))
    return f_alpha, f_delta


def eccentric_anomaly(t, orb_T_0, orb_P, orb_e):
    """
    Calculates eccentric anomaly for the orbit at time t.
    
    The functional form for eccentric anomaly needs to be solved numerically. 
    However, the function converges fast, thus with a good starting point, 
    a simple iterative method with only a few iterations is sufficient to reach
    accurate values.
    
    Parameters
    ----------
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    
    orb_P: Orbital period, in unit of days.
    
    orb_e: Orbital eccentricity
    
    Returns
    -------
    E_obs: Eccentric anomaly at time t as an array, in radians.

    """
    M = 2 * np.pi * (t - orb_T_0) / orb_P
    E_obs = M + (orb_e * np.sin(M)) + ((orb_e**2) * np.sin(2 * M)/M)
    for solve_iteration in range(10):
        M0 = E_obs - (orb_e * np.sin(E_obs))
        E1 = E_obs + ((M - M0) / (1 - (orb_e * np.cos(E_obs))))
        E_obs = E1
    return E_obs


def true_anomaly(t, orb_T_0, orb_P, orb_e):
    """
    Calculates true anomaly for the orbit at time t
    
    Parameters
    ----------
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    
    orb_P: Orbital period, in unit of days.
    
    orb_e: Orbital eccentricity
    
    Returns
    -------
    theta_obs: True anomaly at time t as an array, in radians.
    
    """
    E_obs = eccentric_anomaly(t, orb_T_0, orb_P, orb_e)
    tan_theta2 = np.sqrt( (1 + orb_e) / (1 - orb_e)) * np.tan(E_obs / 2.0) # This is tan(theta/2), where theta is the true anomaly
    orbphase = (E_obs / 2.0) % (2 * np.pi)
    quadrant = (orbphase < (np.pi / 2.0)) | (orbphase > (3.0 * np.pi / 2.0))
    theta_obs = np.ndarray(len(E_obs))
    for obs_orbphase in range(len(orbphase)):
        if quadrant[obs_orbphase]:
            theta_obs[obs_orbphase] = (2 * np.arctan(tan_theta2[obs_orbphase]))
        else:
            theta_obs[obs_orbphase] = (2 * (np.arctan(tan_theta2[obs_orbphase]) + np.pi) )
    return theta_obs


def orbital_motion(t, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a):
    """
    Calculates projected components of the orbital motion on the sky, x for in RA, y for in Dec
    
    Parameters
    ----------
    t: time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  
    
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    
    orb_P: Orbital period, in unit of days.
    
    orb_e: Orbital eccentricity, unitless.
    
    orb_i: Orbital inclination to the LOS, in degrees.
    
    orb_omega: The argument of periastron, in degrees.
    
    orb_Omega: The longitude of the ascending node, in degrees.
    
    orb_a: The orbit size in milliarcseconds on the sky, recommended to be in degrees.
    
    Returns
    -------    
    x_obs, y_obs: Components of the orbital motion on the sky, in RA and Dec directions respectively, in the same units as orb_a.
    
    """
    E_obs = eccentric_anomaly(t, orb_T_0, orb_P, orb_e)
    theta_obs = true_anomaly(t, orb_T_0, orb_P, orb_e)
    orb_omega_rad = orb_omega * np.pi/180.0
    orb_Omega_rad = orb_Omega * np.pi/180.0
    orb_i_rad = orb_i * np.pi/180.0
    x_obs = orb_a * (1 - orb_e * np.cos(E_obs)) * ((np.cos(theta_obs + orb_omega_rad) * np.sin(orb_Omega_rad)) + (np.sin(theta_obs + orb_omega_rad) * np.cos(orb_Omega_rad) * np.cos(orb_i_rad)))
    y_obs = orb_a * (1 - orb_e * np.cos(E_obs)) * ((np.cos(theta_obs + orb_omega_rad) * np.cos(orb_Omega_rad)) - (np.sin(theta_obs + orb_omega_rad) * np.sin(orb_Omega_rad) * np.cos(orb_i_rad)))
    return x_obs, y_obs


def total_motion(t, alpha, delta, alpha_0, delta_0, pm_alpha, pm_delta, parallax, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a, binary_orbit=True):
    """
    Main model function estimating projected motion on the sky:

    Parameters
    ----------
    t: Observation Time
       an array in MJD format and Barycentric Dynamical Time (TDB) scale.  

    alpha: Observation RA, an array, in degrees.
    
    delta: Observation Dec, an array, in degrees.
    
    alpha_0: Expected source RA at time t_0, in degrees.
    
    delta_0: Expected source Dec at time t_0, in degrees.
    
    pm_ra: Proper motion in RA, in milliarcsec/year.
    
    pm_dec: Proper motion in Dec, in milliarcsec/year.
    
    parallax: Source parallax in milliarcsec.
    
    orb_T_0: Time of superior conjunction (when the BH is behind the star), in unit of days.
    
    orb_P: Orbital period, in unit of days.
    
    orb_e: Orbital eccentricity, unitless.
    
    orb_i: Orbital inclination to the LOS, in degrees.
    
    orb_omega: The argument of periastron, in degrees.
    
    orb_Omega: The longitude of the ascending node, in degrees.
    
    orb_a: The orbit size in milliarcseconds on the sky, in milliarcsec.
    
    binary_orbit: (Bool) whether impact of the binary motion on sky projection should be considered
    
    Returns
    -------    
    RA_model, Dec_model: Expected RA and Dec for given t, in degrees.
    
    """
    # Converting some input values to units consistent with other functions used here:
    pm_alpha_deg = pm_alpha / 3.6e6 / 365.25     # Converting proper motion from milliarcsec/yr to degree/day
    pm_delta_deg = pm_delta / 3.6e6 / 365.25     # Converting proper motion from milliarcsec/yr to degree/day
    orb_a_deg = orb_a / 3.6e6                    # Converting orbit size from milliarcsec to degree
    parallax_deg = parallax / 3.6e6              # Converting parallax from milliarcsec to degree
    if binary_orbit:
        x_obs, y_obs = orbital_motion(t, orb_T_0, orb_P, orb_e, orb_i, orb_omega, orb_Omega, orb_a_deg)
    else:
        x_obs, y_obs = 0.0, 0.0
    frac_alpha, frac_delta = frac_parallax(t, alpha, delta)
    predict_ra = alpha_0 * np.cos(radians(delta)) + (pm_alpha_deg * (t - t_0(t))) + frac_alpha * parallax_deg + x_obs
    predict_dec = delta_0 + (pm_delta_deg * (t - t_0(t))) + frac_delta * parallax_deg + y_obs
    return predict_ra, predict_dec


def jet_coordinates(alpha, delta, beta, alpha_er, delta_er):
    """
    Applying a rotation matrix to transform ra and dec into parallel and perpendicular 
    to jet axis components.
    
    Parameters
    ----------
    alpha: Observation RA, an array, in degrees.
    
    delta: Observation Dec, an array, in degrees.

    beta: rotation angle (jet angle), in degrees.

    alpha_er: Uncertainty in observation RA, an array, in degrees.
    
    delta_er: Uncertainty in observation Dec, an array, in degrees.
    
    Returns
    -------    
    theta_perp, theta_para, theta_perp_er, theta_para_er: rotated coordinates, 
    perpendicular and parallel to the jet, and their uncertainties, respectively (all in degrees).
    
    """
    beta_radian = np.radians(beta)
    theta_perp = alpha * np.cos(beta_radian) + delta * np.sin(beta_radian)
    theta_para = delta * np.cos(beta_radian) - alpha * np.sin(beta_radian)
    theta_perp_er = np.sqrt((alpha_er * np.cos(beta_radian))**2 + (delta_er * np.sin(beta_radian))**2)
    return theta_perp, theta_para


#jet_angle = 26.0
#theta_perp_t, theta_para_t = jet_coordinates(observation_alpha * np.cos(np.radians(observation_delta)), observation_delta, jet_angle)
#theta_perp_er = np.sqrt((observation_alpha_obs_er * np.cos(np.radians(jet_angle)))**2 + (observation_delta_obs_er * np.sin(np.radians(jet_angle)))**2)
