import json
import geopandas as gpd
import pandas as pd
import asyncio
from erclient.client import AsyncERClient
from ecoscope.io.utils import to_hex

import ecoscope


def fatal_status_code(e):
    return 400 <= e.response.status_code < 500


class AsyncEarthRangerIO(AsyncERClient):
    def __init__(self, sub_page_size=4000, tcp_limit=5, prefetch_display_names=False, **kwargs):
        if "server" in kwargs:
            server = kwargs.pop("server")
            kwargs["service_root"] = f"{server}/api/v1.0"
            kwargs["token_url"] = f"{server}/oauth2/token"

        self.sub_page_size = sub_page_size
        self.tcp_limit = tcp_limit
        self.event_type_display_values = {}

        kwargs["client_id"] = kwargs.get("client_id", "das_web_client")
        super().__init__(**kwargs)

        if prefetch_display_names:
            asyncio.get_event_loop().run_until_complete(self._get_display_map())

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

    async def _get_sources_generator(
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

        params = self._clean_kwargs(
            addl_kwargs,
            manufacturer_id=manufacturer_id,
            provider_key=provider_key,
            provider=provider,
            id=id,
            page_size=4000,
        )

        async for source in self._get_data("sources/", params=params):
            yield source

    async def _get_sources_dataframe(self, **kwargs):
        sources = []
        async for source in self._get_sources_generator(**kwargs):
            sources.append(source)

        return pd.DataFrame(sources)

    def get_sources(self, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._get_sources_dataframe(**kwargs))

    async def _get_subjects_generator(
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
            page_size=4000,
        )

        assert params.get("subject_group") is None or params.get("group_name") is None

        if params.get("group_name") is not None:
            try:
                params["subject_group"] = await self._get_data(
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

        async for source in self._get_data("subjects/", params=params):
            yield source

    async def _get_subjects_dataframe(self, **kwargs):
        subjects = []
        async for subject in self._get_subjects_generator(**kwargs):
            subjects.append(subject)

        df = pd.DataFrame(subjects)
        assert not df.empty
        df["hex"] = df["additional"].str["rgb"].map(to_hex) if "additional" in df else "#ff0000"

        return df

    def get_subjects(self, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._get_subjects_dataframe(**kwargs))

    async def _get_subjectsources_generator(self, subjects=None, sources=None, **addl_kwargs):
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

        async for subjectsource in self._get_data("subjectsources/", params=params):
            yield subjectsource

    async def _get_subjectsources_dataframe(self, **kwargs):
        subject_sources = []
        async for subject_source in self._get_subjectsources_generator(**kwargs):
            subject_sources.append(subject_source)

        df = pd.DataFrame(subject_sources)
        return df

    async def get_patrol_types_dataframe(self):
        response = await self._get("activity/patrols/types", params={})

        df = pd.DataFrame(response)
        df.set_index("id", inplace=True)
        return df

    async def _get_patrols_generator(self, since=None, until=None, patrol_type=None, status=None, **addl_kwargs):
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

        async for patrol in self._get_data("activity/patrols", params=params):
            yield patrol

    async def _get_observations_generator(
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
            page_size=4000,
        )

        if source_ids:
            id_name, ids = "source_id", source_ids
        elif subject_ids:
            id_name, ids = "subject_id", subject_ids
        else:
            id_name, ids = "subjectsource_id", subjectsource_ids

        params[id_name] = ids
        async for observation in self._get_data(object="observations/", params=params):
            yield observation

    async def _get_observations_gdf(self, **kwargs):
        observations = []
        async for observation in self._get_observations_generator(**kwargs):
            observations.append(observation)

        observations = pd.DataFrame(observation)

        if observations.empty:
            return gpd.GeoDataFrame()

        observations["created_at"] = pd.to_datetime(
            observations["created_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(kwargs.get("tz"))

        observations["recorded_at"] = pd.to_datetime(
            observations["recorded_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(kwargs.get("tz"))

        observations.sort_values("recorded_at", inplace=True)
        return AsyncEarthRangerIO._to_gdf(observations)

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
        observations = ecoscope.base.Relocations()
        df_pt = None

        if include_patrol_details:
            df_pt = await self.get_patrol_types_dataframe()

        tasks = []
        async for patrol in self._get_patrols_generator(
            since=since, until=until, patrol_type=patrol_type, status=status
        ):
            task = asyncio.create_task(self._get_observations_by_patrol(patrol, relocations, tz, df_pt, **kwargs))
            tasks.append(task)

        observations = await asyncio.gather(*tasks)
        observations = pd.concat(observations)

        return observations

    async def _get_observations_by_patrol(self, patrol, relocations=True, tz="UTC", patrol_types=None, **kwargs):

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
                    subject_id=subject_id, start=patrol_start_time, end=patrol_end_time, **kwargs
                ):
                    observations_by_subject.append(obs)

                observations_by_subject = pd.DataFrame(observations_by_subject)
                if observations_by_subject.empty:
                    continue

                observations_by_subject["subject_id"] = subject_id
                observations_by_subject["created_at"] = pd.to_datetime(
                    observations_by_subject["created_at"],
                    errors="coerce",
                    utc=True,
                ).dt.tz_convert(tz)

                observations_by_subject["recorded_at"] = pd.to_datetime(
                    observations_by_subject["recorded_at"],
                    errors="coerce",
                    utc=True,
                ).dt.tz_convert(tz)
                observations_by_subject.sort_values("recorded_at", inplace=True)
                observations_by_subject = AsyncEarthRangerIO._to_gdf(observations_by_subject)

                # TODO - handle include flag requests
                #  include_source_details
                #  include_subject_details
                #  include_subjectsource_details

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
                    observations_by_subject.set_index("id", inplace=True)

                if len(observations_by_subject) > 0:
                    observations = pd.concat([observations, observations_by_subject])

            except Exception as e:
                print(
                    f"Getting observations for subject_id={subject_id} start_time={patrol_start_time}"
                    f"end_time={patrol_end_time} failed for: {e}"
                )
        return observations

    async def _get_event_types_generator(self, include_inactive=False, **addl_kwargs):
        params = self._clean_kwargs(addl_kwargs, include_inactive=include_inactive)

        response = await self._get("activity/events/eventtypes", params=params)
        for obj in response:
            yield obj

    async def get_event_schema(self, event_type_name):
        return await self._get(f"activity/events/schema/eventtype/{event_type_name}", params={})

    async def _get_display_map(self):
        async def _store_event_props(event_type_name):
            schema = await self.get_event_schema(event_type_name)
            self.event_type_display_values.get(event_type_name)["properties"] = dict(
                [(k, v["title"]) for k, v in schema["schema"]["properties"].items()]
            )

        tasks = []
        async for event_type in self._get_event_types_generator():
            self.event_type_display_values[event_type["value"]] = {"display": event_type["display"]}
            task = asyncio.create_task(_store_event_props(event_type_name=event_type["value"]))
            tasks.append(task)
        await asyncio.gather(*tasks)
