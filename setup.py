import os
import re

from setuptools import setup

version_file = open("ecoscope/version.py").read()
version_data = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', version_file))

try:
    descr = open(os.path.join(os.path.dirname(__file__), "README.rst")).read()
except IOError:
    descr = ""

dependencies = [
    "affine",
    "astroplan",
    "backoff",
    "datashader",
    "earthengine-api",
    "earthranger-client",
    "folium",
    "geopandas<=0.14.2",
    "igraph",
    "ipywidgets",
    "kaleido",
    "mapclassify",
    "matplotlib",
    "networkx",
    "numba",
    "plotly",
    "pyarrow",
    "pyproj",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-reportlog",
    "rasterio",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "selenium",
    "tqdm",
    "xyzservices",
]

setup(
    name="ecoscope",
    version=version_data["version"],
    description="Standard Analytical Reporting Framework for Conservation",
    long_description=descr,
    url="http://github.com/wildlife-dynamics/ecoscope",
    author="Jake Wall",
    author_email="walljcg@gmail.com",
    license="MIT",
    packages=[
        "ecoscope",
        "ecoscope.analysis",
        "ecoscope.analysis.UD",
        "ecoscope.base",
        "ecoscope.contrib",
        "ecoscope.io",
        "ecoscope.mapping",
        "ecoscope.plotting",
    ],
    platforms="Posix; MacOS X; Windows",
    install_requires=dependencies,
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
)
