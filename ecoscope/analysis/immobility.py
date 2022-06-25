import datetime
import typing
from dataclasses import dataclass

import ecoscope


@dataclass
class ImmobilityProfile:
    """
    threshold_radius: radius of cluster in metres.  Defaults to 13m.
    threshold_time: time in seconds the track is expected to be stationary. Defaults to 18000 seconds (5 hours)
    threshold_probability: the proportion of observations in a sample which
    must be inside a cluster to generate a CRITICAL.  Default 0.8

    """

    threshold_radius: int = 13
    threshold_time: int = 18000
    threshold_probability: float = 0.8


class Immobility:
    @classmethod
    def calculate_immobility(
        cls, immobility_profile: ImmobilityProfile, relocs: ecoscope.base.Relocations
    ) -> typing.Dict:
        """
        Function to search for immobility within a movement trajectory. Assumes we start with a filtered
        trajectory spanning some period of time. The algorithm will work backwards through the trajectory's
        relocations and build a cluster. Looks to see if the cluster characteristics match immobility criteria
        (ie., timespan is gte than the threshold_time, and the cluster probability is gte to the threshold_probability)
        Note that this is a simplified version of the full clustering algorithm since it's only looking at data within
        threshold time and will not figure out the true start of an immobility without looking backwards through all
        possible points
        TODO: include more info about the immobility result:
        1) immobility start time
        2) immobility probability
        3) immobility cluster fix count
        4) algorithm provenance

        Parameters
        ----------
        immobility_profile: ImmobilityProfile
            setting for immobility
        relocs: ecoscope.base.Relocations
            set for fixes for given subject.

        Returns
        -------
            typing.Dict

        """

        relocs.remove_filtered(inplace=True)
        relocs.sort_values(by="fixtime", ascending=True, inplace=True)
        ts = relocs.fixtime.iat[-1] - relocs.fixtime.iat[0]

        if ts < datetime.timedelta(seconds=immobility_profile.threshold_time):
            raise Exception("Insufficient Data")

        relocs.sort_values(by="fixtime", ascending=False, inplace=True)

        cluster_pvalue = (
            relocs.threshold_point_count(threshold_dist=immobility_profile.threshold_radius) / relocs.shape[0]
        )

        if (cluster_pvalue >= immobility_profile.threshold_probability) and (
            ts.total_seconds() > immobility_profile.threshold_time
        ):
            return {
                "probability_value": cluster_pvalue,
                "cluster_radius": relocs.cluster_radius,
                "cluster_fix_count": relocs.threshold_point_count(immobility_profile.threshold_radius),
                "total_fix_count": relocs.shape[0],
                "immobility_time": ts,
            }
