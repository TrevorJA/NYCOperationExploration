"""
MRF factor profile generation from seasonal parameterization.

Handles:
- Season boundary shifts
- Season-specific scaling factors
- Reconstruction of 366-day profiles for all reservoir/level combinations
"""

import numpy as np
from typing import Dict

# Default season definitions (from FFMP Table 4c)
# DOY values are 1-indexed (Jan 1 = 1, Dec 31 = 365 or 366)
DEFAULT_SEASONS = {
    "spring": {
        "start_doy": 121,  # May 1
        "end_doy": 151,    # May 31
    },
    "summer": {
        "start_doy": 152,  # Jun 1
        "end_doy": 243,    # Aug 31
    },
    "fall": {
        "start_doy": 244,  # Sep 1
        "end_doy": 334,    # Nov 30
    },
    "winter": {
        "start_doy": 335,  # Dec 1 (wraps to next year)
        "end_doy": 120,    # Apr 30
    }
}

# All MRF factor profile names (3 reservoirs × 7 levels = 21 profiles)
RESERVOIRS = ["cannonsville", "pepacton", "neversink"]
DROUGHT_LEVELS = ["level1a", "level1b", "level1c", "level2", "level3", "level4", "level5"]


def get_all_mrf_profile_names():
    """Return list of all MRF factor profile names."""
    return [
        f"{level}_factor_mrf_{reservoir}"
        for level in DROUGHT_LEVELS
        for reservoir in RESERVOIRS
    ]


class MRFProfileBuilder:
    """
    Build MRF factor profiles with parameterized season boundaries and scales.

    This class applies seasonal modifications (boundary shifts and scaling factors)
    to the base MRF factor profiles. The same modifications are applied to all
    21 profiles (3 reservoirs × 7 drought levels) since they share the same
    seasonal structure from FFMP Table 4c.
    """

    def __init__(self, base_profiles: Dict[str, np.ndarray]):
        """
        Initialize with default profiles.

        Parameters
        ----------
        base_profiles : dict
            Dictionary mapping profile names to 366-day arrays.
            E.g., {"level2_factor_mrf_cannonsville": np.array([...])}
        """
        self.base_profiles = base_profiles

    def _get_season_mask(self, start_doy: int, end_doy: int) -> np.ndarray:
        """
        Create a boolean mask for days within a season.

        Parameters
        ----------
        start_doy : int
            Start day of year (1-indexed, after applying shifts)
        end_doy : int
            End day of year (1-indexed, after applying shifts)

        Returns
        -------
        np.ndarray
            Boolean mask of shape (366,) where True indicates days in the season
        """
        # Convert to 0-indexed for array operations
        start_idx = (start_doy - 1) % 366
        end_idx = (end_doy - 1) % 366

        days = np.arange(366)

        if start_idx <= end_idx:
            # Normal case: season doesn't wrap around year end
            return (days >= start_idx) & (days <= end_idx)
        else:
            # Wrap-around case (e.g., winter: Dec 1 to Apr 30)
            return (days >= start_idx) | (days <= end_idx)

    def apply_seasonal_modifications(self,
                                     profile_name: str,
                                     season_shifts: Dict[str, int],
                                     season_scales: Dict[str, float]) -> np.ndarray:
        """
        Apply seasonal boundary shifts and scaling to a profile.

        Parameters
        ----------
        profile_name : str
            Profile name, e.g., "level2_factor_mrf_cannonsville"
        season_shifts : dict
            Mapping of season name to shift in days.
            E.g., {"summer": -5, "fall": 0, "winter": 3, "spring": 0}
        season_scales : dict
            Mapping of season name to scaling factor.
            E.g., {"summer": 1.1, "fall": 0.95, "winter": 1.0, "spring": 1.05}

        Returns
        -------
        np.ndarray
            Modified 366-day profile array
        """
        if profile_name not in self.base_profiles:
            raise ValueError(f"Unknown profile: {profile_name}")

        base = self.base_profiles[profile_name].copy()
        modified = np.zeros(366)

        for season_name, season_def in DEFAULT_SEASONS.items():
            shift = season_shifts.get(season_name, 0)
            scale = season_scales.get(season_name, 1.0)

            # Calculate shifted boundaries
            shifted_start = season_def["start_doy"] + shift
            shifted_end = season_def["end_doy"] + shift

            # Get mask for this season's days
            season_mask = self._get_season_mask(shifted_start, shifted_end)

            # Apply scaling to this season's days
            modified[season_mask] = base[season_mask] * scale

        return modified

    def build_all_modified_profiles(self,
                                    season_shifts: Dict[str, int],
                                    season_scales: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Build all modified profiles for all reservoirs and levels.

        Parameters
        ----------
        season_shifts : dict
            Season boundary shifts in days
        season_scales : dict
            Season scaling factors

        Returns
        -------
        dict
            Dictionary of profile_name -> modified 366-day array
        """
        modified = {}
        for profile_name in self.base_profiles:
            if "_factor_mrf_" in profile_name:
                modified[profile_name] = self.apply_seasonal_modifications(
                    profile_name, season_shifts, season_scales
                )
        return modified


def build_season_params_from_sample(param_dict: dict) -> tuple:
    """
    Extract season shifts and scales from a parameter sample dictionary.

    Parameters
    ----------
    param_dict : dict
        Dictionary of parameter names to values from Sobol sampling

    Returns
    -------
    tuple
        (season_shifts, season_scales) dictionaries
    """
    season_shifts = {}
    season_scales = {}

    for season in ["summer", "fall", "winter", "spring"]:
        shift_key = f"mrf_{season}_start_shift"
        scale_key = f"mrf_{season}_scale"

        if shift_key in param_dict:
            season_shifts[season] = int(round(param_dict[shift_key]))
        if scale_key in param_dict:
            season_scales[season] = param_dict[scale_key]

    return season_shifts, season_scales


def has_mrf_profile_params(param_dict: dict) -> bool:
    """Check if parameter dict contains any MRF profile parameters."""
    mrf_profile_params = [
        "mrf_summer_start_shift", "mrf_fall_start_shift",
        "mrf_winter_start_shift", "mrf_spring_start_shift",
        "mrf_summer_scale", "mrf_fall_scale",
        "mrf_winter_scale", "mrf_spring_scale"
    ]
    return any(key in param_dict for key in mrf_profile_params)
