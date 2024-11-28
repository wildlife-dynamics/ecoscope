import json
import geopandas as gpd
import pandas as pd
import asyncio

import ecoscope
from ecoscope.io.utils import to_hex
from ecoscope.io.earthranger_utils import clean_kwargs, to_gdf, clean_time_cols
from erclient.client import ERClientException, ERClientNotFound

try:
    from erclient.client import AsyncERClient
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["async_earthranger"]'
    )


class AsyncEarthRangerIO(AsyncERClient):
    def __init__(self, sub_page_size=4000, tcp_limit=5, **kwargs):
        if "server" in kwargs:
            server = kwargs.pop("server")
            kwargs["service_root"] = f"{server}/api/v1.0"
            kwargs["token_url"] = f"{server}/oauth2/token"

        self.sub_page_size = sub_page_size
        self.tcp_limit = tcp_limit
        self.event_type_display_values = None

        kwargs["client_id"] = kwargs.get("client_id", "das_web_client")
        super().__init__(**kwargs)

    @classmethod
    async def create(cls, **kwargs):
        client = cls(**kwargs)

        await client._init_client()
        return client

    async def _init_client(self):
        if not self.auth:
            try:
                await self.login()
            except ERClientNotFound:
                raise ERClientNotFound("Failed login. Check Stack Trace for specific reason.")
        else:
            try:
                await self.get_me()
            except ERClientException:
                raise ERClientException("Authorization token is invalid or expired.")

    async def get_me(self):
        return await self._get("user/me", params={})

    async def get_sources(
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
        providers
        id
        Returns
        -------
        An async generator to iterate over sources.
        """

        params = clean_kwargs(
            addl_kwargs,
            manufacturer_id=manufacturer_id,
            provider_key=provider_key,
            provider=provider,
            id=id,
            page_size=addl_kwargs.pop("page_size", self.sub_page_size),
        )

        async for source in self._get_data("sources/", params=params):
            yield source

    async def get_sources_dataframe(
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
        providers
        id
        Returns
        -------
        sources : pd.DataFrame
            DataFrame of queried sources
        """
        sources = []
        async for source in self.get_sources(**addl_kwargs):
            sources.append(source)

        df = pd.DataFrame(sources)
        df = clean_time_cols(df)
        return df

    async def get_subjects(
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
        An async generator to iterate over subjects
        """

        params = clean_kwargs(
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
            page_size=addl_kwargs.pop("page_size", self.sub_page_size),
        )

        assert params.get("subject_group") is None or params.get("group_name") is None

        if params.get("group_name") is not None:
            try:
                subject_groups = await self._get(
                    "subjectgroups/",
                    params={
                        "group_name": params.pop("group_name"),
                        "include_inactive": True,
                        "include_hidden": True,
                        "flat": True,
                    },
                )
                params["subject_group"] = subject_groups[0]["id"]
            except IndexError:
                raise KeyError("`group_name` not found")

        async for source in self._get_data("subjects/", params=params):
            yield source

    async def get_subjects_dataframe(
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
        subjects = []
        async for subject in self.get_subjects(**addl_kwargs):
            subjects.append(subject)

        df = pd.DataFrame(subjects)

        if not df.empty:
            df["hex"] = df["additional"].str["rgb"].map(to_hex) if "additional" in df else "#ff0000"
            df = clean_time_cols(df)
        return df

    async def get_subjectsources(self, subjects=None, sources=None, **addl_kwargs):
        """
        Parameters
        ----------
        subjects: A comma-delimited list of Subject IDs.
        sources: A comma-delimited list of Source IDs.
        Returns
        -------
        An async generator to iterate over subjectsources
        """
        params = clean_kwargs(addl_kwargs, sources=sources, subjects=subjects)

        async for subjectsource in self._get_data("subjectsources/", params=params):
            yield subjectsource

    async def get_subjectsources_dataframe(self, subjects=None, sources=None, **addl_kwargs):
        """
        Parameters
        ----------
        subjects: A comma-delimited list of Subject IDs.
        sources: A comma-delimited list of Source IDs.
        Returns
        -------
        subjectsources : pd.DataFrame
        """
        subject_sources = []
        async for subject_source in self.get_subjectsources(**addl_kwargs):
            subject_sources.append(subject_source)

        df = pd.DataFrame(subject_sources)
        df = clean_time_cols(df)
        return df

    async def get_patrol_types_dataframe(self):
        response = await self._get("activity/patrols/types", params={})

        df = pd.DataFrame(response)
        df.set_index("id", inplace=True)
        return df

    async def get_patrols(self, since=None, until=None, patrol_type=None, status=None, **addl_kwargs):
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

        params = clean_kwargs(
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

        async for patrol in self._get_data("activity/patrols", params=params):
            yield patrol

    async def get_patrols_dataframe(self, **kwargs):
        patrols = []
        async for patrol in self.get_patrols(**kwargs):
            patrols.append(patrol)

        df = pd.DataFrame(patrols)
        df = clean_time_cols(df)
        return df

    async def get_observations(
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

        params = clean_kwargs(
            addl_kwargs,
            since=since,
            until=until,
            filter=filter,
            include_details=include_details,
            created_after=created_after,
            page_size=addl_kwargs.pop("page_size", self.sub_page_size),
        )

        if source_ids:
            id_name, ids = "source_id", source_ids
        elif subject_ids:
            id_name, ids = "subject_id", subject_ids
        else:
            id_name, ids = "subjectsource_id", subjectsource_ids

        params[id_name] = ids
        async for observation in self._get_data("observations/", params=params):
            yield observation

    async def get_observations_gdf(self, **kwargs):
        observations = []
        async for observation in self.get_observations(**kwargs):
            observations.append(observation)

        observations = pd.DataFrame(observation)

        if observations.empty:
            return gpd.GeoDataFrame()
        observations = clean_time_cols(observations)
        observations["created_at"] = observations["created_at"].dt.tz_convert(kwargs.get("tz"))
        observations["recorded_at"] = observations["recorded_at"].dt.tz_convert(kwargs.get("tz"))

        observations.sort_values("recorded_at", inplace=True)
        return to_gdf(observations)

    async def get_patrol_observations_with_patrol_filter(
        self,
        since=None,
        until=None,
        patrol_type=None,
        status=None,
        include_patrol_details=False,
        relocations=True,
        tz="UTC",
        **kwargs,
    ):
        """
        Download observations for patrols with provided filters.
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
        include_patrol_details : bool, optional
            Whether to merge patrol details into dataframe
        kwargs
            Additional parameters to pass to `_get_observations_by_patrol`.
        Returns
        -------
        relocations : ecoscope.base.Relocations
        """
        observations = ecoscope.base.Relocations()
        df_pt = await self.get_patrol_types_dataframe() if include_patrol_details else None

        tasks = []
        async for patrol in self.get_patrols(since=since, until=until, patrol_type=patrol_type, status=status):
            task = asyncio.create_task(self._get_observations_by_patrol(patrol, relocations, tz, df_pt, **kwargs))
            tasks.append(task)

        observations = await asyncio.gather(*tasks)
        observations = pd.concat(observations)

        if include_patrol_details:
            observations["groupby_col"] = observations["patrol_id"]

        return observations

    async def _get_observations_by_patrol(self, patrol, relocations=True, tz="UTC", patrol_types=None, **kwargs):
        """
        Download observations by patrol.
        Parameters
        ----------
        patrol:
            The patrol to download observations for
        relocations:
            If true, returns a ecoscope.base.Relocations object instead of a GeoDataFrame
        tz:
            The timezeone to return observation times in
        patrol_types:
            Comma-separated list of type of patrol UUID
        kwargs:
            Additional parameters to pass to `get_observations`.
        Returns
        -------
        relocations : ecoscope.base.Relocations
        """
        observations = ecoscope.base.Relocations()
        for patrol_segment in patrol["patrol_segments"]:
            subject_id = (patrol_segment.get("leader") or {}).get("id")
            patrol_start_time = (patrol_segment.get("time_range") or {}).get("start_time")
            patrol_end_time = (patrol_segment.get("time_range") or {}).get("end_time")

            if None in {subject_id, patrol_start_time}:
                continue

            try:
                observations_by_subject = []
                async for obs in self.get_observations(
                    subject_ids=subject_id, since=patrol_start_time, until=patrol_end_time, **kwargs
                ):
                    observations_by_subject.append(obs)

                observations_by_subject = pd.DataFrame(observations_by_subject)
                if observations_by_subject.empty:
                    continue

                observations_by_subject["subject_id"] = subject_id

                observations_by_subject = clean_time_cols(observations_by_subject)
                observations_by_subject["created_at"] = observations_by_subject["created_at"].dt.tz_convert(tz)
                observations_by_subject["recorded_at"] = observations_by_subject["recorded_at"].dt.tz_convert(tz)
                observations_by_subject = to_gdf(observations_by_subject)

                if relocations:
                    observations_by_subject = ecoscope.base.Relocations.from_gdf(
                        observations_by_subject,
                        groupby_col="subject_id",
                        uuid_col="id",
                        time_col="recorded_at",
                    )

                if patrol_types is not None:
                    patrol_type = patrol_types[
                        patrol_types["value"] == patrol_segment.get("patrol_type")
                    ].reset_index()["id"][0]
                    observations_by_subject["patrol_id"] = patrol["id"]
                    observations_by_subject["patrol_title"] = patrol["title"]
                    observations_by_subject["patrol_serial_number"] = patrol["serial_number"]
                    observations_by_subject["patrol_start_time"] = patrol_start_time
                    observations_by_subject["patrol_end_time"] = patrol_end_time
                    observations_by_subject["patrol_type"] = patrol_type
                    observations_by_subject = (
                        observations_by_subject.reset_index()
                        .merge(
                            patrol_types.add_prefix("patrol_type__"),
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
                    observations_by_subject = clean_time_cols(observations_by_subject)
                    observations_by_subject.set_index("id", inplace=True)

                if len(observations_by_subject) > 0:
                    observations = pd.concat([observations, observations_by_subject])

            except Exception as e:
                print(
                    f"Getting observations for subject_id={subject_id} start_time={patrol_start_time}"
                    f"end_time={patrol_end_time} failed for: {e}"
                )
        return observations

    async def get_events_dataframe(
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

        params = clean_kwargs(
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
            page_size=addl_kwargs.pop("page_size", 100),
        )
        filter = {"date_range": {}}
        if since is not None:
            filter["date_range"]["lower"] = since
            params["filter"] = json.dumps(filter)
        if until is not None:
            filter["date_range"]["upper"] = until
            params["filter"] = json.dumps(filter)

        events = []
        async for event in self.get_events(**params):
            events.append(event)

        df = pd.DataFrame(events)
        df = clean_time_cols(df)
        gdf = gpd.GeoDataFrame(df)
        if gdf.loc[0, "location"] is not None:
            gdf.loc[~gdf["geojson"].isna(), "geometry"] = gpd.GeoDataFrame.from_features(
                gdf.loc[~gdf["geojson"].isna(), "geojson"]
            )["geometry"]
            gdf.set_geometry("geometry", inplace=True)
            gdf.set_crs(4326, inplace=True)

        gdf.sort_values("time", inplace=True)
        gdf.set_index("id", inplace=True)
        return gdf

    async def get_event_types(self, include_inactive=False, **addl_kwargs):
        """
        Parameters
        ----------
        include_inactive: Include inactive subjects in list.
        **addl_kwargs: Additional query params
        Returns
        -------
        An async generator to iterate over event types
        """
        params = clean_kwargs(addl_kwargs, include_inactive=include_inactive)

        response = await self._get("activity/events/eventtypes", params=params)
        for obj in response:
            yield obj

    async def get_event_schema(self, event_type_name):
        """
        Parameters
        ----------
        event_type_name: The event type to fetch
        Returns
        -------
        The event type schema json
        """
        return await self._get(f"activity/events/schema/eventtype/{event_type_name}", params={})

    async def load_display_map(self):
        """
        Loads event type display values into event_type_display_values
        """
        self.event_type_display_values = {}

        async def _store_event_props(event_type_name):
            schema = await self.get_event_schema(event_type_name)
            self.event_type_display_values.get(event_type_name)["properties"] = dict(
                [(k, v["title"]) for k, v in schema["schema"]["properties"].items()]
            )

        tasks = []
        async for event_type in self.get_event_types():
            self.event_type_display_values[event_type["value"]] = {"display": event_type["display"]}
            task = asyncio.create_task(_store_event_props(event_type_name=event_type["value"]))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def get_event_type_display_name(self, event_type, event_property=None):
        """
        Parameters
        ----------
        event_type: The event type name to fetch the display value of
        event_property: If provided, returns the display name of the provided event property
        Returns
        -------
        """
        if self.event_type_display_values is None:
            await self.load_display_map()

        if event_property is not None:
            return self.event_type_display_values.get(event_type).get("properties").get(event_property)

        return self.event_type_display_values.get(event_type).get("display")
