from __future__ import annotations

from itertools import chain
from typing import List, Union

import numpy as np
import scipy


class VideoTracklet:
    """A collection of video observations for an individual."""

    def __init__(self, track_id: Union[int, List[int]], embeds: List[np.ndarray] = []):
        """Initializes a video tracklet.

        Args:
                track_id: track of list of tracks in this tracklet
                embeds: center data for the track of shape [n_videos, embed_dim]
        """
        self._track_id = track_id if isinstance(track_id, list) else [track_id]
        assert len(embeds.shape) == 2
        self._embeddings = embeds

    @property
    def track_id(self):
        """List of ids contained in this tracklet."""
        return self._track_id

    @property
    def embeddings(self):
        """Embedding data for this tracklet."""
        return self._embeddings

    @classmethod
    def from_tracklets(cls, tracklets: List[VideoTracklet]):
        """Builds a new tracklet from 1 or more tracklets.

        Args:
                tracklets:
                        List of VideoTracklet objects to join together
        """
        all_track_ids = list(chain.from_iterable([x.track_id for x in tracklets]))
        all_embeddings = np.concatenate([x.embeddings for x in tracklets])
        return cls(all_track_ids, all_embeddings)

    @staticmethod
    def cosine_distance(
        first: VideoTracklet, second: VideoTracklet, default_val: float = np.nan
    ):
        """Calculates the cosine distance between two tracklets.

        Args:
                first: first tracklet to compare
                second: second tracklet to compare
                default_val: value supplied if distance invalid

        Returns:
                Returns the smallest cosine distance between the two tracklets.
                Since tracklets can contain multiple centers, any pair can result in this minimum.
                If either tracklet has invalid embedding data, default_val.
        """
        embeds_1 = first.embeddings
        embeds_2 = second.embeddings

        if embeds_1.shape[0] == 0 or embeds_2.shape[0] == 0:
            return default_val

        distance = scipy.spatial.distance.cdist(embeds_1, embeds_2, metric="cosine")
        return np.min(distance)

    def compare_to(self, other: VideoTracklet):
        """Compares this tracklet with another."""
        return self.cosine_distance(self, other)


class Fragment:
    """A collection of tracklets that overlap in time."""

    def __init__(self, tracklets: List[VideoTracklet]):
        """Initializes a fragment object.

        Args:
                tracklets: List of tracklets
        """
        self._tracklets = tracklets
        self._separation = self._calculate_mean_separation()

    @property
    def separation(self):
        """Average separation between tracklets."""
        return self._separation

    def _calculate_mean_separation(self):
        """Calculates the mean cosine distance between contained tracklets."""
        separations = []
        for i in np.arange(len(self._tracklets) - 1):
            for j in np.arange(len(self._tracklets) - 1 - i) + i + 1:
                distance = VideoTracklet.cosine_distance(
                    self._tracklets[i], self._tracklets[j]
                )
                if np.isnan(distance):
                    separations.append(0.0)
                else:
                    separations.append(distance)
        return np.mean(separations)

    def hungarian_match_ids(self, other: Fragment):
        """Matches this fragment with another through hungarian matching."""
        distance_cost = np.zeros(
            [len(self._tracklets), len(other._tracklets)], dtype=np.float64
        )
        for i in np.arange(len(self._tracklets)):
            for j in np.arange(len(other._tracklets)):
                distance_cost[i, j] = VideoTracklet.cosine_distance(
                    self._tracklets[i], other._tracklets[j]
                )
        if np.all(np.isnan(distance_cost)):
            row_best, col_best = ([], [])
        else:
            row_best, col_best = scipy.optimize.linear_sum_assignment(distance_cost)
        return row_best, col_best
