import json
import geopandas as gpd
import pandas as pd
from dateutil import parser
from erclient.client import AsyncERClient

import ecoscope
from ecoscope.io.utils import to_hex


def fatal_status_code(e):
    return 400 <= e.response.status_code < 500


class AsyncEarthRangerIO(AsyncERClient):
    def __init__(self, sub_page_size=4000, tcp_limit=5, **kwargs):
        if "server" in kwargs:
            server = kwargs.pop("server")
            kwargs["service_root"] = f"{server}/api/v1.0"
            kwargs["token_url"] = f"{server}/oauth2/token"

        self.sub_page_size = sub_page_size
        self.tcp_limit = tcp_limit
        kwargs["client_id"] = kwargs.get("client_id", "das_web_client")
        super().__init__(**kwargs)
        # try:
        #     self.login()
        # except ERClientNotFound:
        #     raise ERClientNotFound("Failed login. Check Stack Trace for specific reason.")

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

    async def get_events_df(
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

        df = pd.DataFrame()
        async for res in self.get_events(params):
            df.concat(pd.DataFrame(res))

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

        df = pd.DataFrame()

        async for res in self._get_data("subjects/", params=params):
            df.concat(pd.DataFrame(res))

        assert not df.empty

        df["hex"] = df["additional"].str["rgb"].map(to_hex) if "additional" in df else "#ff0000"

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

    async def get_patrol_observations(
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

        """patrols struct
        {'id': 'ea2b9c29-a9f1-4a32-9634-962690d96618',
         'priority': 0,
         'state': 'done',
         'objective': None,
         'serial_number': 14150,
         'title': 'End of foot patrol starting motorbike back to camp',
         'files': [],
         'notes': [],
         'patrol_segments': [{'id': '751a5d54-42dc-4c8b-9416-68d77577d32a', 'patrol_type': 'mwaluganje_routine_motorbike_patrol', 'leader': {'content_type': 'observations.subject', 'id': 'e146c97b-912b-430c-95b2-4e9c3a2186c6', 'name': 'Suleiman Kibo', 'subject_type': 'person', 'subject_subtype': 'ranger', 'common_name': None, 'additional': {'rgb': '200, 70, 146', 'sex': '', 'region': 'Rangers-MEP', 'country': 'Kenya', 'external_id': '', 'tm_animal_id': '', 'external_name': ''}, 'created_at': '2024-01-23T10:38:53.337249+03:00', 'updated_at': '2024-06-30T10:50:30.734207+03:00', 'is_active': True, 'user': {'id': '0b6a95fa-3295-4e78-9b3a-9aee19ae5a48'}, 'region': 'Rangers-MEP', 'country': 'Kenya', 'sex': '', 'tracks_available': False, 'image_url': '/static/ranger-black.svg'}, 'scheduled_start': None, 'scheduled_end': None, 'time_range': {'start_time': '2024-05-24T11:01:06+03:00', 'end_time': '2024-05-24T11:33:00+03:00'}, 'start_location': {'latitude': -4.0904833, 'longitude': 39.465905}, 'end_location': {'latitude': -4.074973, 'longitude': 39.4845189}, 'events': [], 'image_url': 'https://mep.pamdas.org/static/sprite-src/traffic_rep.svg', 'icon_id': 'traffic_rep', 'updates': [{'message': 'Updated fields: End Time', 'time': '2024-05-25T12:51:12.164228+00:00', 'user': {'username': 'ckagume', 'first_name': 'Caroline', 'last_name': 'Mumbi', 'id': '06c5ea42-5041-4cf1-9360-d556dddca3e2', 'content_type': 'accounts.user'}, 'type': 'update_segment'}, {'message': 'Updated fields: End Time, End Location', 'time': '2024-05-24T08:39:09.737928+00:00', 'user': {'username': 'karani', 'first_name': 'Suleiman', 'last_name': 'Kibo', 'id': '0b6a95fa-3295-4e78-9b3a-9aee19ae5a48', 'content_type': 'accounts.user'}, 'type': 'update_segment'}]}], 'updates': [{'message': 'Updated fields: State is done', 'time': '2024-05-24T08:39:09.757917+00:00', 'user': {'username': 'karani', 'first_name': 'Suleiman', 'last_name': 'Kibo', 'id': '0b6a95fa-3295-4e78-9b3a-9aee19ae5a48', 'content_type': 'accounts.user'}, 'type': 'update_patrol_state'}, {'message': 'Patrol Added', 'time': '2024-05-24T08:01:28.516621+00:00', 'user': {'username': 'karani', 'first_name': 'Suleiman', 'last_name': 'Kibo', 'id': '0b6a95fa-3295-4e78-9b3a-9aee19ae5a48', 'content_type': 'accounts.user'}, 'type': 'add_patrol'}]}
        """  # noqa

        observations = []
        # ignoring patrol types for now
        # patrol_types = await self._get_data("activity/patrols/types", params={})

        async for patrol in self.get_patrols(since=since, until=until, patrol_type=patrol_type, status=status):
            for patrol_segment in patrol["patrol_segments"]:
                subject_id = (patrol_segment.get("leader") or {}).get("id")
                patrol_start_time = (patrol_segment.get("time_range") or {}).get("start_time")
                patrol_end_time = (patrol_segment.get("time_range") or {}).get("end_time")

                # ignoring patrol types for now
                # patrol_type = df_pt[df_pt["value"] == patrol_segment.get("patrol_type")].reset_index()["id"][0]

                if None in {subject_id, patrol_start_time}:
                    continue

                try:
                    observations_by_subject = []
                    async for obs in self.get_observations(
                        subject_id=subject_id, start=patrol_start_time, end=patrol_end_time, **kwargs
                    ):
                        observations_by_subject.append(obs)

                    observations_by_subject = pd.DataFrame(observations_by_subject)
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
                    observations_by_subject = self._to_gdf(observations_by_subject)

                    if relocations:
                        observations_by_subject = ecoscope.base.Relocations.from_gdf(
                            observations_by_subject,
                            groupby_col="subject_id",
                            uuid_col="id",
                            time_col="recorded_at",
                        )

                    # TODO re-add handling for patrol details
                    # if include_patrol_details:
                    #     observation["patrol_id"] = patrol["id"]
                    #     observation["patrol_serial_number"] = patrol["serial_number"]
                    #     observation["patrol_start_time"] = patrol_start_time
                    #     observation["patrol_end_time"] = patrol_end_time
                    #     observation["patrol_type"] = patrol_type
                    #     observation = (
                    #         observation.reset_index()
                    #         .merge(
                    #             pd.DataFrame(df_pt).add_prefix("patrol_type__"),
                    #             left_on="patrol_type",
                    #             right_on="id",
                    #         )
                    #         .drop(
                    #             columns=[
                    #                 "patrol_type__ordernum",
                    #                 "patrol_type__icon_id",
                    #                 "patrol_type__default_priority",
                    #                 "patrol_type__is_active",
                    #             ]
                    #         )
                    #     )

                    if len(observations_by_subject) > 0:
                        observations.append(observations_by_subject)

                except Exception as e:
                    print(
                        f"Getting observations for subject_id={subject_id} start_time={patrol_start_time}"
                        f"end_time={patrol_end_time} failed for: {e}"
                    )

        df = ecoscope.base.Relocations(pd.DataFrame(observations))
        # if include_patrol_details:
        #     return df.set_index("id")
        return df

    async def get_subject_observations(
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

        observations = []
        async for obs in self._get_observations(subject_ids=subject_ids, **kwargs):
            observations.append(obs)

        if observations.empty:
            return gpd.GeoDataFrame()

        # if include_source_details:
        #     observations = observations.merge(
        #         self.get_sources(id=",".join(observations["source"].unique())).add_prefix("source__"),
        #         left_on="source",
        #         right_on="source__id",
        #     )
        # if include_subject_details:
        #     if isinstance(subject_ids, pd.DataFrame):
        #         observations = observations.merge(
        #             subject_ids.add_prefix("subject__"),
        #             left_on="subject_id",
        #             right_on="subject__id",
        #         )
        #     else:
        #         observations = observations.merge(
        #             self.get_subjects(id=",".join(subject_ids), include_inactive=True).add_prefix("subject__"),
        #             left_on="subject_id",
        #             right_on="subject__id",
        #         )

        # if include_subjectsource_details:
        #     observations = observations.merge(
        #         self.get_subjectsources(subjects=",".join(observations["subject_id"].unique())).add_prefix(
        #             "subjectsource__"
        #         ),
        #         left_on=["subject_id", "source"],
        #         right_on=["subjectsource__subject", "subjectsource__source"],
        #     )

        if relocations:
            return ecoscope.base.Relocations.from_gdf(
                observations,
                groupby_col="subject_id",
                uuid_col="id",
                time_col="recorded_at",
            )
        else:
            return observations
