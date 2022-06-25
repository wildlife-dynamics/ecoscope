import concurrent.futures
import datetime
import itertools
import math
import re
import typing

import backoff
import geopandas as gpd
import pandas as pd
import requests
from tqdm.auto import tqdm

import ecoscope
from ecoscope.contrib.dasclient import DasClient, DasClientException
from ecoscope.io.utils import pack_columns, to_hex


class BackoffDasClient(DasClient):
    """
    Extends DasClient to add backoff to requests.

    """

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=10,
    )
    @backoff.on_exception(
        backoff.expo,
        DasClientException,
        max_tries=10,
        giveup=lambda e: len(e.args) == 0
        or not isinstance(e.args[0], str)
        or re.match("Failed to call DAS web service. 50[24] .*", e.args[0]) is None,
    )
    def _get(self, *args, **kwargs):
        return super()._get(*args, **kwargs)


class ConcurrentDasClient(BackoffDasClient):
    """
    Extends DasClient to make multiple requests while maintaining interface.

    """

    def __init__(self, *, sub_page_size=4000, tcp_limit=5, **kwargs):
        self.sub_page_size = sub_page_size
        self.tcp_limit = tcp_limit
        super().__init__(**kwargs)

    def _concurrent_get(self, path, stream=False, **kwargs):
        page_size = kwargs.get("params", {}).get("page_size")

        if page_size is None or page_size % self.sub_page_size or stream:
            return super()._get(path, stream=stream, **kwargs)

        p = kwargs["params"].copy()
        p["page"] = int(p.get("page", 1))
        current_page = p["page"]

        p["page_size"] = 1
        response = super()._get(path=path, params=p)
        p["page_size"] = self.sub_page_size

        if "count" not in response:
            return response

        if not response.get("count"):
            return response

        start = 1 + (p["page"] - 1) * page_size // self.sub_page_size
        end = 1 + min(
            p["page"] * page_size // self.sub_page_size,
            math.ceil(response["count"] / self.sub_page_size),
        )

        if response["previous"] is not None:
            response["previous"] = response["previous"].replace("page_size=1", f"page_size={page_size}")
        if response["count"] > page_size * current_page:
            response["next"] = response["next"].replace("page_size=1", f"page_size={page_size}")
        else:
            response["next"] = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.tcp_limit) as executor:
            futures = []
            for page in range(start, end):
                p["page"] = page
                futures.append(executor.submit(super()._get, path=path, params=p.copy()))

            response["results"] = itertools.chain(
                *(future.result()["results"] for future in concurrent.futures.as_completed(futures))
            )
        return response

    def _get(self, *args, **kwargs):
        params = kwargs.get("params", {})
        if "sub_page_size" in params or "tcp_limit" in params:
            print(
                f"Warning: `sub_page_size` and `tcp_limit` should only be provided to the constructor of {type(self)}"
            )
        params["page"] = params.get("page", 1)
        params["page_size"] = params.get("page_size", 1000000000)
        params = {k: v if not isinstance(v, bool) else str(v).lower() for k, v in params.items()}
        kwargs["params"] = params
        return self._concurrent_get(*args, **kwargs)


class EarthRangerIO(ConcurrentDasClient):
    """
    Extends ConcurrentDasClient with Ecoscope-specific configuration.

    """

    def __init__(self, **kwargs):
        if "server" in kwargs:
            server = kwargs.pop("server")
            kwargs["service_root"] = server + "/api/v1.0"
            kwargs["token_url"] = server + "/oauth2/token"

        kwargs["client_id"] = kwargs.get("client_id", "das_web_client")

        super().__init__(**kwargs)

    @staticmethod
    def _to_gdf(df):
        longitude, latitude = (0, 1) if isinstance(df["location"].iat[0], list) else ("longitude", "latitude")
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["location"].str[longitude], df["location"].str[latitude]),
            crs=4326,
        )

    @staticmethod
    def _normalize_column(df, col):
        print(col)
        for k, v in pd.json_normalize(df.pop(col), sep="__").add_prefix(f"{col}__").iteritems():
            df[k] = v.values

    @staticmethod
    def _clean_kwargs(addl_kwargs={}, **kwargs):
        for k in addl_kwargs.keys():
            print(f"Warning: {k} is a non-standard parameter. Results may be unexpected.")
        return {k: v for k, v in {**addl_kwargs, **kwargs}.items() if v is not None}

    def get_subjectsources(self, subjects=None, sources=None, **addl_kwargs):
        """
        Parameters
        ----------
        subjects
            A comma-delimited list of Subject IDs.
        sources
            A comma-delimited list of Source IDs.

        Returns
        -------
        subjectsources : pd.DataFrame
            DataFrame of queried subjectssources

        """

        kwargs = self._clean_kwargs(
            addl_kwargs,
            subjects=subjects,
            sources=sources,
        )

        return pd.DataFrame(self._get("subjectsources/", params=kwargs)["results"])

    def get_subjects(
        self,
        include_inactive=None,
        tracks_since=None,
        tracks_until=None,
        bbox=None,
        subject_group=None,
        name=None,
        updated_since=None,
        render_last_location=None,
        tracks=None,
        id=None,
        updated_until=None,
        group_name=None,
        **addl_kwargs,
    ):
        """
        Parameters
        ----------
        include_inactive
            Include inactive subjects in list.
        tracks_since
            Include tracks since this timestamp
        tracks_until
            Include tracks up through this timestamp
        bbox
            Include subjects having track data within this bounding box defined by a 4-tuple of coordinates marking
            west, south, east, north.
        subject_group
            Indicate a subject group for which Subjects should be listed.
        name : UUID
            Find subjects with the given name.
        updated_since
            Return Subject that have been updated since the given timestamp.
        render_last_location
            Indicate whether to render each subject's last location.
        tracks
            Indicate whether to render each subject's recent tracks.
        id
            A comma-delimited list of Subject IDs.
        updated_until
        group_name

        Returns
        -------
        subjects : pd.DataFrame
            DataFrame of queried subjects

        """

        kwargs = self._clean_kwargs(
            addl_kwargs,
            include_inactive=include_inactive,
            tracks_since=tracks_since,
            tracks_until=tracks_until,
            bbox=bbox,
            subject_group=subject_group,
            name=name,
            updated_since=updated_since,
            render_last_location=render_last_location,
            tracks=tracks,
            id=id,
            updated_until=updated_until,
            group_name=group_name,
        )

        assert kwargs.get("subject_group") is None or kwargs.get("group_name") is None

        if kwargs.get("group_name") is not None:
            try:
                kwargs["subject_group"] = self._get(
                    "subjectgroups/",
                    params={
                        "group_name": kwargs.pop("group_name"),
                        "include_inactive": True,
                        "include_hidden": True,
                        "flat": True,
                    },
                )[0]["id"]
            except IndexError:
                raise KeyError("`group_name` not found")

        df = pd.DataFrame(self._get("subjects/", params=kwargs)["results"])
        assert not df.empty

        df["hex"] = df["additional"].str["rgb"].map(to_hex) if "additional" in df else "#ff0000"

        return df

    def get_patrols(self, filter=None, status=None, **addl_kwargs):
        """
        Parameters
        ----------
        filter:
            example: {"date_range":{"lower":"2020-09-16T00:00:00.000Z"}}
            date_range
            patrols_overlap_daterange
            text
            patrol_type
            tracked_by
        status
            Comma-separated list of 'scheduled'/'active'/'overdue'/'done'/'cancelled'

        Returns
        -------
        patrols : pd.DataFrame
            DataFrame of queried patrols

        """

        kwargs = self._clean_kwargs(
            addl_kwargs,
            filter=filter,
            status=status,
        )

        return pd.DataFrame(self._get("activity/patrols/", params=kwargs)["results"])

    def get_users(self) -> pd.DataFrame:
        """
        Get all users

        Returns
        -------
        users: pd.DataFrame

        """
        return pd.DataFrame(super().get_users())

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

        kwargs = self._clean_kwargs(
            addl_kwargs,
            manufacturer_id=manufacturer_id,
            provider_key=provider_key,
            provider=provider,
            id=id,
        )

        return pd.DataFrame(self._get("sources/", params=kwargs)["results"])

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
        subject_ids
            filter to a single subject
        source_ids
            filter to a single source
        subjectsource_ids
            filter to a subjectsource_id, rather than source_id + time range
        since
            get observations after this ISO8061 date, include timezone
        until
            get observations up to this ISO8061 date, include timezone
        filter
            filter using exclusion_flags for an observation.
            filter=None returns everything
            filter=0  filters out everything but rows with exclusion flag 0 (i.e, passes back clean data)
            filter=1  filters out everything but rows with exclusion flag 1 (i.e, passes back manually filtered data)
            filter=2, filters out everything but rows with exclusion flag 2 (i.e., passes back automatically filtered
            data)
            filter=3, filters out everything but rows with exclusion flag 2 or 1 (i.e., passes back both manual and
            automatically filtered data)
        include_details
            one of [true,false], default is false. This brings back the observation additional field
        created_after
            get observations created (saved in EarthRanger) after this ISO8061 date, include timezone

        Returns
        -------
        observations : gpd.GeoDataFrame
            GeoDataFrame of queried observations

        """

        assert (source_ids, subject_ids, subjectsource_ids).count(None) == 2

        kwargs = self._clean_kwargs(
            addl_kwargs,
            since=since,
            until=until,
            filter=filter,
            include_details=include_details,
            created_after=created_after,
        )

        if source_ids is not None:
            id_name = "source_id"
            ids = source_ids
        elif subject_ids is not None:
            id_name = "subject_id"
            ids = subject_ids
        else:
            id_name = "subjectsource_id"
            ids = subjectsource_ids

        if isinstance(ids, str):
            ids = [ids]

        observations = []
        pbar = tqdm(ids)
        for id in pbar:
            pbar.set_description(f"Downloading Observations for {id_name}={id}")
            kwargs[id_name] = id
            df = pd.DataFrame(self._get("observations/", params=kwargs)["results"])
            df[id_name] = id
            observations.append(df)

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
        subject_ids : str or list[str]
            List of subject UUIDs
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

        observations = self._get_observations(subject_ids=subject_ids, **kwargs)

        if observations.empty:
            return gpd.GeoDataFrame()

        if include_source_details:
            observations = observations.merge(
                pd.DataFrame(self.get_sources(id=",".join(observations["source"].unique()))).add_prefix("source__"),
                left_on="source",
                right_on="source__id",
            )

        if include_subject_details:
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

    def get_events(
        self,
        is_collection=None,
        updated_size=None,
        event_ids=None,
        bbox=None,
        include_updates=None,
        include_details=None,
        sort_by=None,
        patrol_segment=None,
        state=None,
        event_type=None,
        include_notes=None,
        include_related_events=None,
        include_files=None,
        max_results=None,
        oldest_update_date=None,
        exclude_contained=None,
        updated_since=None,
        event_category=None,
        filter=None,
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
        include_updates
            Boolean value
        include_details
            Boolean value
        sort_by
            Sort by (use 'event_time', 'updated_at', 'created_at', 'serial_number') with optional minus ('-') prefix to
            reverse order.
        patrol_segment
            ID of patrol segment to filter on
        state
            Comma-separated list of 'scheduled'/'active'/'overdue'/'done'/'cancelled'
        event_type
            Comma-separated list of event type uuids
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
        filter : json dict
            Can contain any of 'event_filter_id', 'text', 'date_range', 'duration', 'state', 'priority',
            'event_category', 'event_type', 'reported_by', 'create_date', 'update_date'

        Returns
        -------
        events : gpd.GeoDataFrame
            GeoDataFrame of queried events

        """

        kwargs = self._clean_kwargs(
            addl_kwargs,
            is_collection=is_collection,
            updated_size=updated_size,
            event_ids=event_ids,
            bbox=bbox,
            include_updates=include_updates,
            include_details=include_details,
            sort_by=sort_by,
            patrol_segment=patrol_segment,
            state=state,
            event_type=event_type,
            include_notes=include_notes,
            include_related_events=include_related_events,
            include_files=include_files,
            max_results=max_results,
            oldest_update_date=oldest_update_date,
            exclude_contained=exclude_contained,
            updated_since=updated_since,
            event_category=event_category,
            filter=filter,
        )

        df = pd.DataFrame(self._get("activity/events/", params=kwargs)["results"])
        assert not df.empty
        df["time"] = pd.to_datetime(df["time"])

        df = gpd.GeoDataFrame(df)
        df.loc[~df["geojson"].isna(), "geometry"] = gpd.GeoDataFrame.from_features(
            df.loc[~df["geojson"].isna(), "geojson"]
        )["geometry"]
        df.set_crs(4326, inplace=True)

        df.sort_values("time", inplace=True)
        return df

    def get_subjectgroup_observations(self, subject_group=None, group_name=None, include_inactive=True, **kwargs):
        """
        Parameters
        ----------
        subject_group : str
            UUID of subject group to filter by
        group_name : str
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

        assert (subject_group is None) != (group_name is None)

        if subject_group:
            subject_ids = self.get_subjects(subject_group=subject_group, include_inactive=include_inactive).id.tolist()
        else:
            subject_ids = self.get_subjects(group_name=group_name, include_inactive=include_inactive).id.tolist()

        return self.get_subject_observations(subject_ids, **kwargs)

    def get_observations_for_patrols(self, patrols_df, **kwargs):
        """
        Download observations for provided `patrols_df`.

        Parameters
        ----------
        patrols_df : pd.DataFrame
            Data returned from a call to `get_patrols`.
        kwargs
            Additional parameters to pass to `get_subject_observations`.

        Returns
        -------
        relocations : ecoscope.base.Relocations

        """

        observations = []
        for _, patrol in patrols_df.iterrows():
            for patrol_segment in patrol["patrol_segments"]:
                subject_id = (patrol_segment.get("leader") or {}).get("id")
                start_time = (patrol_segment.get("time_range") or {}).get("start_time")
                end_time = (patrol_segment.get("time_range") or {}).get("end_time")

                if None in {subject_id, start_time}:
                    continue
                try:
                    observations.append(
                        self.get_subject_observations(
                            subject_ids=[subject_id],
                            since=start_time,
                            until=end_time,
                            **kwargs,
                        )
                    )
                except Exception as e:
                    # print(f'Getting observations for {subject_id=} {start_time=} {end_time=} failed for: {e}')
                    print(
                        f"Getting observations for subject_id={subject_id} start_time={start_time} end_time={end_time}"
                        f"failed for: {e}"
                    )
        return ecoscope.base.Relocations(pd.concat(observations))

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
            except DasClientException as exc:
                self.logger.error(exc)
            except requests.exceptions.RequestException as exc:
                self.logger.error(exc)
            else:
                return pd.DataFrame(results)

        return observations.groupby(source_id_col).progress_apply(upload)

    def post_subjectsource(
        self,
        subject_id: str,
        source_id: str,
        lower_bound_assignend_range: datetime.datetime,
        upper_bound_assigned_range: datetime.datetime,
        additional: typing.Dict = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        subject_id
        source_id
        lower_bound_assignend_range
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
                "lower": lower_bound_assignend_range,
                "upper": upper_bound_assigned_range,
            },
            "additional": additional,
        }

        urlpath = f"subject/{subject_id}/sources"
        response = self._post(urlpath, payload=payload)
        return pd.DataFrame([response])

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
