from math import ceil, floor

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import sklearn.base
from affine import Affine
from shapely.geometry import shape
from skimage.draw import line

import ecoscope


class Ecograph:
    """
    A class that analyzes movement tracking data using Network Theory.

    Parameters
    ----------
    trajectory : ecoscope.base.Trajectory
        Trajectory dataframe
    resolution : float
        Pixel size, in meters
    radius : int
        Radius to compute Collective Influence (Default : 2)
    cutoff : int
        Cutoff to compute an approximation of betweenness index if the standard algorithm is
        too slow. Can be useful for very large graphs (Default : None)
    tortuosity_length : int
        The number of steps used to compute the two tortuosity metrics (Default : 3 steps)
    """

    def __init__(self, trajectory, resolution=15, radius=2, cutoff=None, tortuosity_length=3):
        self.graphs = {}
        self.trajectory = trajectory
        self.resolution = ceil(resolution)

        self.utm_crs = trajectory.estimate_utm_crs()
        self.trajectory.to_crs(self.utm_crs, inplace=True)
        self.features = [
            "dot_product",
            "speed",
            "step_length",
            "sin_time",
            "cos_time",
            "weight",
            "degree",
            "betweenness",
            "collective_influence",
            "tortuosity_1",
            "tortuosity_2",
        ]
        geom = self.trajectory["geometry"]

        eastings = np.array([geom.iloc[i].coords.xy[0] for i in range(len(geom))]).flatten()
        northings = np.array([geom.iloc[i].coords.xy[1] for i in range(len(geom))]).flatten()

        self.xmin = floor(np.min(eastings)) - self.resolution
        self.ymin = floor(np.min(northings)) - self.resolution
        self.xmax = ceil(np.max(eastings)) + self.resolution
        self.ymax = ceil(np.max(northings)) + self.resolution

        self.xmax += self.resolution - ((self.xmax - self.xmin) % self.resolution)
        self.ymax += self.resolution - ((self.ymax - self.ymin) % self.resolution)

        self.transform = Affine(self.resolution, 0.00, self.xmin, 0.00, -self.resolution, self.ymax)
        self.inverse_transform = ~self.transform

        self.n_rows = int((self.xmax - self.xmin) // self.resolution)
        self.n_cols = int((self.ymax - self.ymin) // self.resolution)

        def compute(df):
            subject_name = df.name
            print(f"Computing EcoGraph for subject {subject_name}")
            G = self._get_ecograph(self, df, subject_name, radius, cutoff, tortuosity_length)
            self.graphs[subject_name] = G

        self.trajectory.groupby("groupby_col").progress_apply(compute)

    def to_csv(self, output_path):
        """
        Saves the features of all nodes in a CSV file

        Parameters
        ----------
        output_path : str, Pathlike
            Output path for the CSV file
        """

        features_id = ["individual_name", "grid_id"] + self.features
        df = {feat_id: [] for feat_id in features_id}
        for individual_name, G in self.graphs.items():
            for node in G.nodes():
                df["individual_name"].append(individual_name)
                df["grid_id"].append(node)
                for feature in self.features:
                    df[feature].append(G.nodes[node][feature])
        (pd.DataFrame.from_dict(df)).to_csv(output_path, index=False)

    def to_geotiff(self, feature, output_path, individual="all", interpolation=None, transform=None):
        """
        Saves a specific node feature as a GeoTIFF

        Parameters
        ----------
        feature : str
            Feature of interest
        output_path : str, Pathlike
            Output path for the GeoTIFF file
        individual : str
            The individual for which we want to output the node feature (Default : "all")
        interpolation : str or None
            Whether to interpolate the feature for each step in the trajectory (Default : None). If
            provided, has to be one of those four types of interpolation : "mean", "median", "max"
            or "min"
        transform : sklearn.base.TransformerMixin or None
            A feature transform method (Default : None)
        """

        if feature in self.features:
            if individual == "all":
                feature_ndarray = self._get_feature_mosaic(self, feature, interpolation)
            elif individual in self.graphs.keys():
                feature_ndarray = self._get_feature_map(self, feature, individual, interpolation)
            else:
                raise IndividualNameError("This individual is not in the dataset")
        else:
            raise FeatureNameError("This feature was not computed by EcoGraph")

        if isinstance(transform, sklearn.base.TransformerMixin):
            nan_mask = ~np.isnan(feature_ndarray)
            feature_ndarray[nan_mask] = transform.fit_transform(feature_ndarray[nan_mask].reshape(-1, 1)).reshape(
                feature_ndarray[nan_mask].shape
            )

        raster_profile = ecoscope.io.raster.RasterProfile(
            pixel_size=self.resolution,
            crs=self.utm_crs,
            nodata_value=np.nan,
            band_count=1,
        )
        raster_profile.raster_extent = ecoscope.io.raster.RasterExtent(
            x_min=self.xmin, x_max=self.xmax, y_min=self.ymin, y_max=self.ymax
        )
        ecoscope.io.raster.RasterPy.write(
            ndarray=feature_ndarray,
            fp=output_path,
            **raster_profile,
        )

    @staticmethod
    def _get_ecograph(self, trajectory_gdf, individual_name, radius, cutoff, tortuosity_length):
        G = nx.Graph()
        geom = trajectory_gdf["geometry"]
        for i in range(len(geom) - (tortuosity_length - 1)):
            step_attributes = trajectory_gdf.iloc[i]
            lines = [list(geom.iloc[i + j].coords) for j in range(tortuosity_length)]
            p1, p2, p3, p4 = lines[0][0], lines[0][1], lines[1][1], lines[1][0]
            pixel1, pixel2 = (
                self.inverse_transform * p1,
                self.inverse_transform * p2,
            )

            row1, row2 = floor(pixel1[0]), floor(pixel2[0])
            col1, col2 = ceil(pixel1[1]), ceil(pixel2[1])

            t = step_attributes["segment_start"]
            seconds_in_day = 24 * 60 * 60
            seconds_past_midnight = (t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1000000.0)
            time_diff = pd.to_datetime(
                trajectory_gdf.iloc[i + (tortuosity_length - 1)]["segment_end"]
            ) - pd.to_datetime(t)
            time_delta = time_diff.total_seconds() / 3600.0
            tortuosity_1, tortuosity_2 = self._get_tortuosities(self, lines, time_delta)

            attributes = {
                "dot_product": self._get_dot_product(p1, p2, p3, p4),
                "speed": step_attributes["speed_kmhr"],
                "step_length": step_attributes["dist_meters"],
                "sin_time": np.sin(2 * np.pi * seconds_past_midnight / seconds_in_day),
                "cos_time": np.cos(2 * np.pi * seconds_past_midnight / seconds_in_day),
                "tortuosity_1": tortuosity_1,
                "tortuosity_2": tortuosity_2,
            }

            if G.has_node((row1, col1)):
                self._update_node(G, (row1, col1), attributes)
            else:
                self._initialize_node(G, (row1, col1), attributes)
            if not G.has_node((row2, col2)):
                self._initialize_node(G, (row2, col2), attributes, empty=True)
            if (row1, col1) != (row2, col2):
                G.add_edge((row1, col1), (row2, col2))

        for node in G.nodes():
            for key in attributes.keys():
                G.nodes[node][key] = np.mean(G.nodes[node][key])

        self._compute_network_metrics(self, G, radius, cutoff)
        return G

    @staticmethod
    def _update_node(G, node_id, attributes):
        G.add_node(node_id)
        G.nodes[node_id]["weight"] += 1
        for key, value in attributes.items():
            if value is not None:
                G.nodes[node_id][key].append(value)

    @staticmethod
    def _get_day_night_value(day_night_value):
        if day_night_value == "day":
            return 0
        elif day_night_value == "night":
            return 1

    @staticmethod
    def _initialize_node(G, node_id, attributes, empty=False):
        G.add_node(node_id)
        G.nodes[node_id]["weight"] = 1
        for key, value in attributes.items():
            if empty:
                G.nodes[node_id][key] = []
            else:
                if value is not None:
                    G.nodes[node_id][key] = [value]

    @staticmethod
    def _get_dot_product(x, y, z, w):
        if (floor(y[0]) == floor(w[0])) and (floor(y[1]) == floor(w[1])):
            v = [y[0] - x[0], y[1] - x[1]]
            w = [z[0] - y[0], z[1] - y[1]]
            angle = np.arctan2(w[1], w[0]) - np.arctan2(v[1], v[0])

            while angle <= -np.pi:
                angle = angle + 2 * np.pi
            while angle > np.pi:
                angle = angle - 2 * np.pi
            return np.cos(angle)
        else:
            return None

    @staticmethod
    def _get_tortuosities(self, lines, time_delta):
        point1, point2 = lines[0][0], lines[len(lines) - 1][1]
        beeline_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        total_length = 0
        for i in range(len(lines) - 1):
            point1, point2, point3 = lines[i][0], lines[i][1], lines[i + 1][0]
            if (floor(point2[0]) == floor(point3[0])) and (floor(point2[1]) == floor(point3[1])):
                total_length += np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
            else:
                return None, np.log(time_delta / (beeline_dist**2))
        point1, point2 = lines[len(lines) - 1][0], lines[len(lines) - 1][1]
        total_length += np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        return (beeline_dist / total_length), np.log(time_delta / (beeline_dist**2))

    @staticmethod
    def _compute_network_metrics(self, G, radius, cutoff):
        self._compute_degree(G)
        self._compute_betweenness(G, cutoff)
        self._compute_collective_influence(self, G, radius)

    @staticmethod
    def _compute_degree(G):
        for node in G.nodes():
            G.nodes[node]["degree"] = G.degree[node]

    @staticmethod
    def _compute_collective_influence(self, G, radius):
        for node in G.nodes():
            G.nodes[node]["collective_influence"] = self._get_collective_influence(G, node, radius)

    @staticmethod
    def _get_collective_influence(G, start, radius):
        sub_g = nx.generators.ego_graph(G, start, radius=radius)
        collective_influence = 0
        for n in sub_g.nodes():
            collective_influence += G.degree[n] - 1
        collective_influence -= G.degree[start]
        return (G.degree[start]) * collective_influence

    @staticmethod
    def _compute_betweenness(G, cutoff):
        g = igraph.Graph.from_networkx(G)
        btw_idx = g.betweenness(cutoff=cutoff)
        for v in g.vs:
            node = v["_nx_name"]
            G.nodes[node]["betweenness"] = btw_idx[v.index]

    @staticmethod
    def _get_feature_mosaic(self, feature, interpolation=None):
        features = []
        for individual in self.graphs.keys():
            features.append(self._get_feature_map(self, feature, individual, interpolation))
        mosaic = np.full((self.n_cols, self.n_rows), np.nan)
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                values = []
                for feature_map in features:
                    grid_val = feature_map[i][j]
                    if not np.isnan(grid_val):
                        values.append(grid_val)
                if len(values) >= 2:
                    mosaic[i][j] = np.mean(values)
                elif len(values) == 1:
                    mosaic[i][j] = values[0]
        return mosaic

    @staticmethod
    def _get_feature_map(self, feature, individual, interpolation):
        if interpolation is not None:
            return self._get_interpolated_feature_map(self, feature, individual, interpolation)
        else:
            return self._get_regular_feature_map(self, feature, individual)

    @staticmethod
    def _get_regular_feature_map(self, feature, individual):
        feature_ndarray = np.full((self.n_cols, self.n_rows), np.nan)
        for node in self.graphs[individual].nodes():
            feature_ndarray[node[1]][node[0]] = (self.graphs[individual]).nodes[node][feature]
        return feature_ndarray

    @staticmethod
    def _get_interpolated_feature_map(self, feature, individual, interpolation):
        feature_ndarray = self._get_regular_feature_map(self, feature, individual)
        individual_trajectory = self.trajectory[self.trajectory["groupby_col"] == individual]
        geom = individual_trajectory["geometry"]
        idxs_dict = {}
        for i in range(len(geom)):
            line1 = list(geom.iloc[i].coords)
            p1, p2 = line1[0], line1[1]
            pixel1, pixel2 = self.inverse_transform * p1, self.inverse_transform * p2
            row1, row2 = floor(pixel1[0]), floor(pixel2[0])
            col1, col2 = ceil(pixel1[1]), ceil(pixel2[1])

            rr, cc = line(col1, row1, col2, row2)
            for j in range(len(rr)):
                if np.isnan(feature_ndarray[rr[j], cc[j]]):
                    if (rr[j], cc[j]) in idxs_dict:
                        idxs_dict[(rr[j], cc[j])].append(feature_ndarray[rr[0], cc[0]])
                    else:
                        idxs_dict[(rr[j], cc[j])] = [feature_ndarray[rr[0], cc[0]]]
        for key, value in idxs_dict.items():
            if interpolation == "max":
                feature_ndarray[key[0], key[1]] = np.max(value)
            elif interpolation == "mean":
                feature_ndarray[key[0], key[1]] = np.mean(value)
            elif interpolation == "median":
                feature_ndarray[key[0], key[1]] = np.median(value)
            elif interpolation == "min":
                feature_ndarray[key[0], key[1]] = np.min(value)
            else:
                raise InterpolationError("Interpolation type not supported by EcoGraph")
        return feature_ndarray


class InterpolationError(Exception):
    pass


class IndividualNameError(Exception):
    pass


class FeatureNameError(Exception):
    pass


def get_feature_gdf(input_path):
    """
    Convert a GeoTIFF feature map into a GeoDataFrame

    Parameters
    ----------
    input_path : str, Pathlike
        Input path for the GeoTIFF file
    """
    shapes = []
    with rasterio.open(input_path) as src:
        crs = src.crs.to_wkt()
        data_array = src.read(1).astype(np.float32)
        data_array[data_array == src.nodata] = np.nan
        shapes.extend(rasterio.features.shapes(data_array, transform=src.transform))

    data = {"value": [], "geometry": []}
    for geom, value in shapes:
        if not np.isnan(value):
            data["geometry"].append(shape(geom))
            data["value"].append(value)

    df = pd.DataFrame.from_dict(data)
    return gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
