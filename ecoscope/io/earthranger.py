import datetime
import json
import math
import typing

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
import requests
from dateutil import parser
from erclient.client import ERClient, ERClientException, ERClientNotFound
from tqdm.auto import tqdm

import ecoscope
from ecoscope.io.utils import pack_columns, to_hex


def fatal_status_code(e):
    return 400 <= e.response.status_code < 500


class EarthRangerIO(ERClient):
    def __init__(self, sub_page_size=4000, tcp_limit=5, **kwargs):
        if "server" in kwargs:
            server = kwargs.pop("server")
            kwargs["service_root"] = f"{server}/api/v1.0"
            kwargs["token_url"] = f"{server}/oauth2/token"

        self.sub_page_size = sub_page_size
        self.tcp_limit = tcp_limit
        kwargs["client_id"] = kwargs.get("client_id", "das_web_client")
        super().__init__(**kwargs)
        try:
            self.login()
        except ERClientNotFound:
            raise ERClientNotFound("Failed login. Check Stack Trace for specific reason.")

    def _token_request(self, payload):
        response = requests.post(self.token_url, data=payload)
        if response.ok:
            self.auth = json.loads(response.text)
            expires_in = int(self.auth["expires_in"]) - 5 * 60
            self.auth_expires = pytz.utc.localize(datetime.datetime.utcnow()) + datetime.timedelta(seconds=expires_in)
            return True

        self.auth = None
        self.auth_expires = pytz.utc.localize(datetime.datetime.min)
        raise ERClientNotFound(json.loads(response.text)["error_description"])

    @staticmethod
    def _clean_kwargs(addl_kwargs={}, **kwargs):
        for k in addl_kwargs.keys():
            print(f"Warning: {k} is a non-standard parameter. Results may be unexpected.")
        return {k: v for k, v in {**addl_kwargs, **kwargs}.items() if v is not None}

    @staticmethod
    def _normalize_column(df, col):
        print(col)
        for k, v in pd.json_normalize(df.pop(col), sep="__").add_prefix(f"{col}__").items():
            df[k] = v.values

    @staticmethod
    def _dataframe_to_dict(events):
        if isinstance(events, gpd.GeoDataFrame):
            events["location"] = pd.DataFrame({"longitude": events.geometry.x, "latitude": events.geometry.y}).to_dict(
                "records"
            )
            del events["geometry"]

        if isinstance(events, pd.DataFrame):
            events = events.to_dict("records")
        return events

    @staticmethod
    def _to_gdf(df):
        longitude, latitude = (0, 1) if isinstance(df["location"].iat[0], list) else ("longitude", "latitude")
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["location"].str[longitude], df["location"].str[latitude]),
            crs=4326,
        )

    """
    GET Functions
    """

    def get_sources(
        self,
        manufacturer_id=None,
        provider_key=None,
        provider=None,
        id=None,
        **addl_kwargs,
    ):
        """
        Parameters
        ----------
        manufacturer_id
        provider_key
        provider
        id
        Returns
        -------
        sources : pd.DataFrame
            DataFrame of queried sources
        """

        params = self._clean_kwargs(
            addl_kwargs,
            manufacturer_id=manufacturer_id,
            provider_key=provider_key,
            provider=provider,
            id=id,
        )
        df = pd.DataFrame(
            self.get_objects_multithreaded(
                object="sources/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
            )
        )
        assert not df.empty
        return df

    def get_subjects(
        self,
        include_inactive=None,
        bbox=None,
        subject_group_id=None,
        name=None,
        updated_since=None,
        tracks=None,
        id=None,
        updated_until=None,
        subject_group_name=None,
        max_ids_per_request=50,
        **addl_kwargs,
    ):
        """
        Parameters
        ----------
        include_inactive: Include inactive subjects in list.
        bbox: Include subjects having track data within this bounding box defined by a 4-tuple of coordinates marking
            west, south, east, north.
        subject_group_id: Indicate a subject group id for which Subjects should be listed.
            This is translated to the subject_group parameter in the ER backend
        name : Find subjects with the given name
        updated_since: Return Subject that have been updated since the given timestamp.
        tracks: Indicate whether to render each subject's recent tracks.
        id: A comma-delimited list of Subject IDs.
        updated_until
        subject_group_name: A subject group name for which Subjects should be listed.
            This is translated to the group_name parameter in the ER backend
        Returns
        -------
        subjects : pd.DataFrame
        """

        params = self._clean_kwargs(
            addl_kwargs,
            include_inactive=include_inactive,
            bbox=bbox,
            subject_group=subject_group_id,
            name=name,
            updated_since=updated_since,
            tracks=tracks,
            id=id,
            updated_until=updated_until,
            group_name=subject_group_name,
        )

        assert params.get("subject_group") is None or params.get("group_name") is None

        if params.get("group_name") is not None:
            try:
                params["subject_group"] = self._get(
                    "subjectgroups/",
                    params={
                        "group_name": params.pop("group_name"),
                        "include_inactive": True,
                        "include_hidden": True,
                        "flat": True,
                    },
                )[0]["id"]
            except IndexError:
                raise KeyError("`group_name` not found")

        if params.get("id") is not None:
            params["id"] = params.get("id").split(",")

            def partial_subjects(subjects):
                params["id"] = ",".join(subjects)
                return pd.DataFrame(
                    self.get_objects_multithreaded(
                        object="subjects/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
                    )
                )

            df = pd.concat(
                [
                    partial_subjects(s)
                    for s in np.array_split(params["id"], math.ceil(len(params["id"]) / max_ids_per_request))
                ],
                ignore_index=True,
            )

        else:
            df = pd.DataFrame(
                self.get_objects_multithreaded(
                    object="subjects/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
                )
            )

        assert not df.empty

        df["hex"] = df["additional"].str["rgb"].map(to_hex) if "additional" in df else "#ff0000"

        return df

    def get_subjectsources(self, subjects=None, sources=None, **addl_kwargs):
        """
        Parameters
        ----------
        subjects: A comma-delimited list of Subject IDs.
        sources: A comma-delimited list of Source IDs.
        Returns
        -------
        subjectsources : pd.DataFrame
        """
        params = self._clean_kwargs(addl_kwargs, sources=sources, subjects=subjects)
        return pd.DataFrame(
            self.get_objects_multithreaded(
                object="subjectsources/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
            )
        )

    def _get_observations(
        self,
        source_ids=None,
        subject_ids=None,
        subjectsource_ids=None,
        tz="UTC",
        since=None,
        until=None,
        filter=None,
        include_details=None,
        created_after=None,
        **addl_kwargs,
    ):
        """
        Return observations matching queries. If `subject_id`, `source_id`, or `subjectsource_id` is specified, the
        index is set to the provided value.
        Parameters
        ----------
        subject_ids: filter to a single subject
        source_ids: filter to a single source
        subjectsource_ids: filter to a subjectsource_id, rather than source_id + time range
        since: get observations after this ISO8061 date, include timezone
        until:get observations up to this ISO8061 date, include timezone
        filter
            filter using exclusion_flags for an observation.
            filter=None returns everything
            filter=0  filters out everything but rows with exclusion flag 0 (i.e, passes back clean data)
            filter=1  filters out everything but rows with exclusion flag 1 (i.e, passes back manually filtered data)
            filter=2, filters out everything but rows with exclusion flag 2 (i.e., passes back automatically filtered
            data)
            filter=3, filters out everything but rows with exclusion flag 2 or 1 (i.e., passes back both manual and
            automatically filtered data)
        include_details: one of [true,false], default is false. This brings back the observation additional field
        created_after: get observations created (saved in EarthRanger) after this ISO8061 date, include timezone
        Returns
        -------
        observations : gpd.GeoDataFrame
        """
        assert (source_ids, subject_ids, subjectsource_ids).count(None) == 2

        params = self._clean_kwargs(
            addl_kwargs,
            since=since,
            until=until,
            filter=filter,
            include_details=include_details,
            created_after=created_after,
        )

        if source_ids:
            id_name, ids = "source_id", source_ids
        elif subject_ids:
            id_name, ids = "subject_id", subject_ids
        else:
            id_name, ids = "subjectsource_id", subjectsource_ids

        observations = []
        pbar = tqdm([ids] if isinstance(ids, str) else ids)

        for _id in pbar:
            params[id_name] = _id
            pbar.set_description(f"Downloading Observations for {id_name}={_id}")
            dataframe = pd.DataFrame(
                self.get_objects_multithreaded(
                    object="observations/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
                )
            )
            dataframe[id_name] = _id
            observations.append(dataframe)

        observations = pd.concat(observations)
        if observations.empty:
            return gpd.GeoDataFrame()

        observations["created_at"] = pd.to_datetime(
            observations["created_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(tz)

        observations["recorded_at"] = pd.to_datetime(
            observations["recorded_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(tz)

        observations.sort_values("recorded_at", inplace=True)
        return EarthRangerIO._to_gdf(observations)

    def get_source_observations(self, source_ids, include_source_details=False, relocations=True, **kwargs):
        """
        Get observations for each listed source and create a `Relocations` object.
        Parameters
        ----------
        source_ids : str or list[str]
            List of source UUIDs
        include_source_details : bool, optional
            Whether to merge source info into dataframe
        kwargs
            Additional arguments to pass in the request to EarthRanger. See the docstring of `_get_observations` for
            info.
        Returns
        -------
        relocations : ecoscope.base.Relocations
            Observations in `Relocations` format
        """

        if isinstance(source_ids, str):
            source_ids = [source_ids]

        observations = self._get_observations(source_ids=source_ids, **kwargs)
        if observations.empty:
            return gpd.GeoDataFrame()

        if include_source_details:
            observations = observations.merge(
                pd.DataFrame(self.get_sources(id=",".join(observations["source"].unique()))).add_prefix("source__"),
                left_on="source",
                right_on="source__id",
            )

        if relocations:
            return ecoscope.base.Relocations.from_gdf(
                observations,
                groupby_col="source",
                uuid_col="id",
                time_col="recorded_at",
            )
        else:
            return observations

    def get_subject_observations(
        self,
        subject_ids,
        include_source_details=False,
        include_subject_details=False,
        include_subjectsource_details=False,
        relocations=True,
        **kwargs,
    ):
        """
        Get observations for each listed subject and create a `Relocations` object.
        Parameters
        ----------
        subject_ids : str or list[str] or pd.DataFrame
            List of subject UUIDs, or a DataFrame of subjects
        include_source_details : bool, optional
            Whether to merge source info into dataframe
        include_subject_details : bool, optional
            Whether to merge subject info into dataframe
        include_subjectsource_details : bool, optional
            Whether to merge subjectsource info into dataframe
        kwargs
            Additional arguments to pass in the request to EarthRanger. See the docstring of `__get_observations` for
            info.
        Returns
        -------
        relocations : ecoscope.base.Relocations
            Observations in `Relocations` format
        """

        if isinstance(subject_ids, str):
            subject_ids = [subject_ids]
        elif isinstance(subject_ids, pd.DataFrame):
            subject_ids = subject_ids.id.tolist()
        elif not isinstance(subject_ids, list):
            raise ValueError(f"subject_ids must be either a str or list[str] or pd.DataFrame, not {type(subject_ids)}")

        observations = self._get_observations(subject_ids=subject_ids, **kwargs)

        if observations.empty:
            return gpd.GeoDataFrame()

        if include_source_details:
            observations = observations.merge(
                self.get_sources(id=",".join(observations["source"].unique())).add_prefix("source__"),
                left_on="source",
                right_on="source__id",
            )
        if include_subject_details:
            if isinstance(subject_ids, pd.DataFrame):
                observations = observations.merge(
                    subject_ids.add_prefix("subject__"),
                    left_on="subject_id",
                    right_on="subject__id",
                )
            else:
                observations = observations.merge(
                    self.get_subjects(id=",".join(subject_ids), include_inactive=True).add_prefix("subject__"),
                    left_on="subject_id",
                    right_on="subject__id",
                )

        if include_subjectsource_details:
            observations = observations.merge(
                self.get_subjectsources(subjects=",".join(observations["subject_id"].unique())).add_prefix(
                    "subjectsource__"
                ),
                left_on=["subject_id", "source"],
                right_on=["subjectsource__subject", "subjectsource__source"],
            )

        if relocations:
            return ecoscope.base.Relocations.from_gdf(
                observations,
                groupby_col="subject_id",
                uuid_col="id",
                time_col="recorded_at",
            )
        else:
            return observations

    def get_subjectsource_observations(
        self,
        subjectsource_ids,
        include_source_details=False,
        relocations=True,
        **kwargs,
    ):
        """
        Get observations for each listed subjectsource and create a `Relocations` object.
        Parameters
        ----------
        subjectsource_ids : str or list[str]
            List of subjectsource UUIDs
        include_source_details : bool, optional
            Whether to merge source info into dataframe
        kwargs
            Additional arguments to pass in the request to EarthRanger. See the docstring of `__get_observations` for
            info.
        Returns
        -------
        relocations : ecoscope.base.Relocations
            Observations in `Relocations` format
        """

        if isinstance(subjectsource_ids, str):
            subjectsource_ids = [subjectsource_ids]

        observations = self._get_observations(subjectsource_ids=subjectsource_ids, **kwargs)

        if observations.empty:
            return gpd.GeoDataFrame()

        if include_source_details:
            observations = observations.merge(
                pd.DataFrame(self.get_sources(id=",".join(observations["source"].unique()))).add_prefix("source__"),
                left_on="source",
                right_on="source__id",
            )

        if relocations:
            return ecoscope.base.Relocations.from_gdf(
                observations,
                groupby_col="subjectsource_id",
                uuid_col="id",
                time_col="recorded_at",
            )
        else:
            return observations

    def get_subjectgroup_observations(
        self, subject_group_id=None, subject_group_name=None, include_inactive=True, **kwargs
    ):
        """
        Parameters
        ----------
        subject_group_id : str
            UUID of subject group to filter by
        subject_group_name : str
            Common name of subject group to filter by
        include_inactive : bool, optional
            Whether to get observations for Subjects marked inactive by EarthRanger
        kwargs
            Additional arguments to pass in the request to `get_subject_observations`. See the docstring of
            `get_subject_observations` for info.
        Returns
        -------
        relocations : ecoscope.base.Relocations
            Observations in `Relocations` format
        """

        assert (subject_group_id is None) != (subject_group_name is None)

        if subject_group_id:
            subjects = self.get_subjects(subject_group_id=subject_group_id, include_inactive=include_inactive)
        else:
            subjects = self.get_subjects(subject_group_name=subject_group_name, include_inactive=include_inactive)

        return self.get_subject_observations(subjects, **kwargs)

    def get_event_types(self, include_inactive=False, **addl_kwargs):
        params = self._clean_kwargs(addl_kwargs, include_inactive=include_inactive)

        return pd.DataFrame(self._get("activity/events/eventtypes", **params))

    def get_events(
        self,
        is_collection=None,
        updated_size=None,
        event_ids=None,
        bbox=None,
        sort_by=None,
        patrol_segment=None,
        state=None,
        event_type=None,
        include_updates=False,
        include_details=False,
        include_notes=False,
        include_related_events=False,
        include_files=False,
        max_results=None,
        oldest_update_date=None,
        exclude_contained=None,
        updated_since=None,
        event_category=None,
        since=None,
        until=None,
        **addl_kwargs,
    ):
        """
        Parameters
        ----------
        is_collection
            true/false whether to filter on is_collection
        updated_since
            date-string to limit on updated_at
        event_ids : array[string]
            Event IDs, comma-separated
        bbox
            bounding box including four coordinate values, comma-separated. Ex. bbox=-122.4,48.4,-122.95,49.0
            (west, south, east, north).
        sort_by
            Sort by (use 'event_time', 'updated_at', 'created_at', 'serial_number') with optional minus ('-') prefix to
            reverse order.
        patrol_segment
            ID of patrol segment to filter on
        state
            Comma-separated list of 'scheduled'/'active'/'overdue'/'done'/'cancelled'
        event_type
            Comma-separated list of event type uuids
        include_updates
            Boolean value
        include_details
            Boolean value
        include_notes
            Boolean value
        include_related_events
            Boolean value
        include_files
            Boolean value
        max_results
        oldest_update_date
        exclude_contained
        event_category
        since
        until
        Returns
        -------
        events : gpd.GeoDataFrame
            GeoDataFrame of queried events
        """

        params = self._clean_kwargs(
            addl_kwargs,
            is_collection=is_collection,
            updated_size=updated_size,
            event_ids=event_ids,
            bbox=bbox,
            sort_by=sort_by,
            patrol_segment=patrol_segment,
            state=state,
            event_type=event_type,
            include_updates=include_updates,
            include_details=include_details,
            include_notes=include_notes,
            include_related_events=include_related_events,
            include_files=include_files,
            max_results=max_results,
            oldest_update_date=oldest_update_date,
            exclude_contained=exclude_contained,
            updated_since=updated_since,
            event_category=event_category,
        )

        filter = {"date_range": {}}
        if since is not None:
            filter["date_range"]["lower"] = since
            params["filter"] = json.dumps(filter)
        if until is not None:
            filter["date_range"]["upper"] = until
            params["filter"] = json.dumps(filter)

        df = pd.DataFrame(
            self.get_objects_multithreaded(
                object="activity/events/", threads=self.tcp_limit, page_size=self.sub_page_size, **params
            )
        )

        assert not df.empty

        df["time"] = df["time"].apply(lambda x: pd.to_datetime(parser.parse(x)))

        gdf = gpd.GeoDataFrame(df)
        if gdf.loc[0, "location"] is not None:
            gdf.loc[~gdf["geojson"].isna(), "geometry"] = gpd.GeoDataFrame.from_features(
                gdf.loc[~gdf["geojson"].isna(), "geojson"]
            )["geometry"]
            gdf.set_crs(4326, inplace=True)

        gdf.sort_values("time", inplace=True)
        return gdf.set_index("id")

    def get_patrol_types(self):
        df = pd.DataFrame(self._get("activity/patrols/types"))
        return df.set_index("id")

    def get_patrols(self, since=None, until=None, patrol_type=None, status=None, **addl_kwargs):
        """
        Parameters
        ----------
        since:
            lower date range
        until:
            upper date range
        patrol_type:
            Comma-separated list of type of patrol UUID
        status
            Comma-separated list of 'scheduled'/'active'/'overdue'/'done'/'cancelled'
        Returns
        -------
        patrols : pd.DataFrame
            DataFrame of queried patrols
        """

        params = self._clean_kwargs(
            addl_kwargs,
            status=status,
            patrol_type=[patrol_type] if isinstance(patrol_type, str) else patrol_type,
            return_data=True,
        )

        filter = {"date_range": {}, "patrol_type": []}

        if since is not None:
            filter["date_range"]["lower"] = since
        if until is not None:
            filter["date_range"]["upper"] = until
        if patrol_type is not None:
            filter["patrol_type"] = params["patrol_type"]
        params["filter"] = json.dumps(filter)

        df = pd.DataFrame(
            self.get_objects_multithreaded(
                object="activity/patrols", threads=self.tcp_limit, page_size=self.sub_page_size, **params
            )
        )
        if "serial_number" in df.columns:
            df = df.sort_values(by="serial_number").reset_index(drop=True)
        return df

    def get_patrol_segments_from_patrol_id(self, patrol_id, **addl_kwargs):
        """
        Download patrols for a given `patrol id`.

        Parameters
        ----------
        patrol_id :
            Patrol UUID.
        kwargs
            Additional parameters to pass to `_get`.

        Returns
        -------
        dataframe : Dataframe of patrols.
        """

        params = self._clean_kwargs(addl_kwargs)

        object = f"activity/patrols/{patrol_id}/"
        df = self._get(object, **params)
        df["patrol_segments"][0].pop("updates")
        df.pop("updates")

        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.items()]))

    def get_patrol_segments(self):
        object = "activity/patrols/segments/"
        return pd.DataFrame(
            self.get_objects_multithreaded(object=object, threads=self.tcp_limit, page_size=self.sub_page_size)
        )

    def get_patrol_observations(self, patrols_df, include_patrol_details=False, **kwargs):
        """
        Download observations for provided `patrols_df`.

        Parameters
        ----------
        patrols_df : pd.DataFrame
           Data returned from a call to `get_patrols`.
        include_patrol_details : bool, optional
           Whether to merge patrol details into dataframe
        kwargs
           Additional parameters to pass to `get_subject_observations`.

        Returns
        -------
        relocations : ecoscope.base.Relocations
        """

        observations = []
        df_pt = self.get_patrol_types()
        for _, patrol in patrols_df.iterrows():
            for patrol_segment in patrol["patrol_segments"]:
                subject_id = (patrol_segment.get("leader") or {}).get("id")
                patrol_start_time = (patrol_segment.get("time_range") or {}).get("start_time")
                patrol_end_time = (patrol_segment.get("time_range") or {}).get("end_time")

                patrol_type = df_pt[df_pt["value"] == patrol_segment.get("patrol_type")].reset_index()["id"][0]

                if None in {subject_id, patrol_start_time}:
                    continue

                try:
                    observation = self.get_subject_observations(
                        subject_ids=[subject_id], since=patrol_start_time, until=patrol_end_time, **kwargs
                    )
                    if include_patrol_details:
                        observation["patrol_id"] = patrol["id"]
                        observation["patrol_serial_number"] = patrol["serial_number"]
                        observation["patrol_start_time"] = patrol_start_time
                        observation["patrol_end_time"] = patrol_end_time
                        observation["patrol_type"] = patrol_type
                        observation = (
                            observation.reset_index()
                            .merge(
                                pd.DataFrame(df_pt).add_prefix("patrol_type__"),
                                left_on="patrol_type",
                                right_on="id",
                            )
                            .drop(
                                columns=[
                                    "patrol_type__ordernum",
                                    "patrol_type__icon_id",
                                    "patrol_type__default_priority",
                                    "patrol_type__is_active",
                                ]
                            )
                        )
                    if len(observation) > 0:
                        observations.append(observation)
                except Exception as e:
                    print(
                        f"Getting observations for subject_id={subject_id} start_time={patrol_start_time}"
                        f"end_time={patrol_end_time} failed for: {e}"
                    )

        df = ecoscope.base.Relocations(pd.concat(observations))
        if include_patrol_details:
            return df.set_index("id")
        return df

    def get_patrol_segment_events(
        self,
        patrol_segment_id=None,
        include_details=False,
        include_files=False,
        include_related_events=False,
        include_notes=False,
        **addl_kwargs,
    ):
        params = self._clean_kwargs(
            addl_kwargs,
            patrol_segment_id=patrol_segment_id,
            include_details=include_details,
            include_files=include_files,
            include_related_events=include_related_events,
            include_notes=include_notes,
        )

        object = f"activity/patrols/segments/{patrol_segment_id}/events/"
        return pd.DataFrame(
            self.get_objects_multithreaded(
                object=object, threads=self.tcp_limit, page_size=self.sub_page_size, **params
            )
        )

    def get_spatial_features_group(self, spatial_features_group_id=None, **addl_kwargs):
        """
        Download spatial features in a spatial features group for a given  `spatial features group id`.

        Parameters
        ----------
        spatial_features_group_id :
            Spatial Features Group UUID.
        kwargs
            Additional parameters to pass to `_get`.

        Returns
        -------
        dataframe : GeoDataFrame of spatial features in a spatial features group.
        """
        params = self._clean_kwargs(addl_kwargs, spatial_features_group_id=spatial_features_group_id)

        object = f"spatialfeaturegroup/{spatial_features_group_id}/"
        spatial_features_group = self._get(object, **params)

        spatial_features = []
        for spatial_feature in spatial_features_group["features"]:
            spatial_features.append(spatial_feature["features"][0])

        return gpd.GeoDataFrame.from_features(spatial_features)

    def get_spatial_feature(self, spatial_feature_id=None, **addl_kwargs):
        """
        Download spatial feature for a given  `spatial feature id`.

        Parameters
        ----------
        spatial_feature_id :
            Spatial Feature UUID.
        kwargs
            Additional parameters to pass to `_get`.

        Returns
        -------
        dataframe : GeoDataFrame of spatial feature.
        """

        params = self._clean_kwargs(addl_kwargs, spatial_feature_id=spatial_feature_id)

        object = f"spatialfeature/{spatial_feature_id}/"
        spatial_feature = self._get(object, **params)
        return gpd.GeoDataFrame.from_features(spatial_feature["features"])

    """
    POST Functions
    """

    def post_source(
        self,
        source_type: str,
        manufacturer_id: str,
        model_name: str,
        provider: str = "default",
        additional: typing.Dict = {},
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        source_type
        manufacturer_id
        model_name
        provider
        additional

        Returns
        -------
        pd.DataFrame
        """

        payload = {
            "source_type": source_type,
            "manufacturer_id": manufacturer_id,
            "model_name": model_name,
            "additional": additional,
            "provider": provider,
        }

        if kwargs:
            payload.update(kwargs)

        response = self._post("sources", payload=payload)
        return pd.DataFrame([response])

    def post_subject(
        self,
        subject_name: str,
        subject_type: str,
        subject_subtype: str,
        is_active: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        subject_name
        subject_subtype
        is_active

        Returns
        -------
        pd.DataFrame
        """

        payload = {
            "name": subject_name,
            "subject_subtype": subject_subtype,
            "is_active": is_active,
        }

        if kwargs:
            payload.update(kwargs)

        response = self._post("subjects", payload=payload)
        return pd.DataFrame([response])

    def post_subjectsource(
        self,
        subject_id: str,
        source_id: str,
        lower_bound_assigned_range: datetime.datetime,
        upper_bound_assigned_range: datetime.datetime,
        additional: typing.Dict = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        subject_id
        source_id
        lower_bound_assigned_range
        upper_bound_assigned_range
        additional

        Returns
        -------
        pd.DataFrame
        """

        if additional is None:
            additional = {}
        payload = {
            "source": source_id,
            "assigned_range": {
                "lower": lower_bound_assigned_range,
                "upper": upper_bound_assigned_range,
            },
            "additional": additional,
        }

        urlpath = f"subject/{subject_id}/sources"
        response = self._post(urlpath, payload=payload)
        return pd.DataFrame([response])

    def post_observations(
        self,
        observations: gpd.GeoDataFrame,
        source_id_col: str = "source",
        recorded_at_col: str = "recorded_at",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        observations : gpd.GeoDataFrame
            observation data to be uploaded
        source_id_col : str
            The source column in the observation dataframe
        recorded_at_col : str
            The observation recorded time column in the dataframe

        Returns
        -------
        None
        """

        def upload(obs):
            try:
                obs = obs.rename(columns={source_id_col: "source", recorded_at_col: "recorded_at"})
                obs["location"] = pd.DataFrame({"longitude": obs.geometry.x, "latitude": obs.geometry.y}).to_dict(
                    "records"
                )
                del obs["geometry"]
                obs = pack_columns(obs, columns=["source", "recorded_at", "location"])
                post_data = obs.to_dict("records")
                results = super(EarthRangerIO, self).post_observation(post_data)
            except ERClientException as exc:
                self.logger.error(exc)
            except requests.exceptions.RequestException as exc:
                self.logger.error(exc)
            else:
                return pd.DataFrame(results)

        return observations.groupby(source_id_col, group_keys=False).progress_apply(upload)

    def post_event(
        self,
        events: typing.Union[gpd.GeoDataFrame, pd.DataFrame, typing.Dict, typing.List[typing.Dict]],
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        events

        Returns
        -------
        pd.DataFrame:
            New events created in EarthRanger.
        """

        events = self._dataframe_to_dict(events)
        results = super().post_event(event=events)
        results = results if isinstance(results, list) else [results]
        return pd.DataFrame(results)

    def post_patrol(self, priority: int, **kwargs) -> pd.DataFrame:
        """
        Parameters
        ----------
        priority

        Returns
        -------
        pd.DataFrame
        """
        payload = {"priority": priority}

        if kwargs:
            payload.update(kwargs)

        response = self._post("activity/patrols", payload=payload)
        return pd.DataFrame([response])

    def post_patrol_segment(
        self,
        patrol_id: str,
        patrol_segment_id: str,
        patrol_type: str = None,
        tracked_subject_id: str = None,
        scheduled_start: str = None,
        scheduled_end: str = None,
        start_time: str = None,
        end_time: str = None,
        start_location: typing.Tuple[float, float] = None,
        end_location: typing.Tuple[float, float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        patrol_id
        patrol_segment_id
        patrol_type
        tracked_subject_id
        scheduled_start
        scheduled_end
        start_time
        end_time
        start_location
        end_location

        Returns
        -------
        pd.DataFrame
        """

        payload = {
            "patrol": patrol_id,
            "patrol_segment": patrol_segment_id,
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_end,
            "time_range": {"start_time": start_time, "end_time": end_time},
        }

        if tracked_subject_id is not None:
            payload.update({"leader": {"content_type": "observations.subject", "id": tracked_subject_id}})
        else:
            payload.update({"leader": None})

        if start_location is not None:
            payload.update({"start_location": {"latitude": start_location[0], "longitude": start_location[1]}})
        else:
            payload.update({"start_location": None})

        if end_location is not None:
            payload.update({"end_location": {"latitude": end_location[0], "longitude": end_location[1]}})
        else:
            payload.update({"end_location": None})

        if patrol_type is not None:
            payload.update({"patrol_type": patrol_type})

        if kwargs:
            payload.update(kwargs)

        response = self._post("activity/patrols/segments/", payload=payload)
        return pd.DataFrame([response])

    def post_patrol_segment_event(
        self,
        patrol_segment_id: str,
        event_type: str,
        **addl_kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        patrol_segment_id
        event_type

        Returns
        -------
        pd.DataFrame
        """

        payload = {
            "patrol_segment": patrol_segment_id,
            "event_type": event_type,
        }

        if addl_kwargs:
            payload.update(addl_kwargs)

        response = self._post(f"activity/patrols/segments/{patrol_segment_id}/events/", payload=payload)
        return pd.DataFrame([response])

    """
    PATCH Functions
    """

    def patch_event(
        self,
        event_id: str,
        events: typing.Union[gpd.GeoDataFrame, pd.DataFrame, typing.Dict, typing.List[typing.Dict]],
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        event_id
            UUID for the event that will be updated.
        events

        Returns
        -------
        pd.DataFrame:
            Updated events in EarthRanger.
        """

        events = self._dataframe_to_dict(events)
        if isinstance(events, list):
            results = [self._patch(f"activity/event/{event_id}", payload=event) for event in events]
        else:
            results = [self._patch(f"activity/event/{event_id}", payload=events)]
        return pd.DataFrame(results)

    """
    DELETE Functions
    """

    def delete_observation(self, observation_id: str):
        """
        Parameters
        ----------
        observation_id
        -------
        """

        self._delete("observation/" + observation_id + "/")
