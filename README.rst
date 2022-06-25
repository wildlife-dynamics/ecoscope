.. image:: ./doc/source/_static/images/logo.svg
   :width: 400
   :height: 200
   :align: center

|PyPI| |Tests| |Codecov|

.. |PyPI| image:: https://img.shields.io/pypi/v/ecoscope.svg
   :target: https://pypi.python.org/pypi/ecoscope

.. |Tests| image:: https://github.com/wildlife-dynamics/ecoscope/workflows/Tests/badge.svg
   :target: https://github.com/wildlife-dynamics/ecoscope/actions?query=workflow%3ATests

.. |Codecov| image:: https://codecov.io/gh/wildlife-dynamics/ecoscope/branch/ECO-118/graphs/badge.svg?token=rXdzfueuhb
   :target: https://codecov.io/gh/wildlife-dynamics/ecoscope

========
Ecoscope
========

Ecoscope is an open-source analysis module for tracking, environmental and conservation data analyses. It provides methods and approaches for: Data I/O (EarthRanger, Google Earth Engine, Landscape Dynamics, MoveBank, Geopandas), Movement Data (Relocations, Trajectories, Home-Ranges, EcoGraph, Recurse, Geofencing, Immobility, Moving Windows, Resampling, Filtering), Visualization (EcoMap, EcoPlot), Environmental analyses (Seasons determination, Remote sensing anomalies), Covariate labeling (Day/Night, Seasonal, GEE Image Collections/Images).

Development & Testing
=====================
Development dependencies are included in `environment.yml`.

Please configure code-quality git hooks with:

.. code:: console

    >>> pre-commit install

Copyright & License
-------------------

Ecoscope is licensed under BSD 3-Clause license. Copyright (c) 2022, wildlife-dynamics.

Donate
------
Ecoscope is currently developed and maintained by the Mara Elephant Project (www.maraelephantproject.org) a not-for-profit elephant conservation organization located in the Masai Mara, Kenya. If you find Ecoscope useful in your work, please consider donating to help support ongoing development and maintenance of the project or the conservation work performed by MEP: https://maraelephantproject.org/donate/

Acknowledgments
---------------
We thank the many individuals and organizations who have contributed to the development of Ecoscope: Jake Wall, Peter Kulits, Zakaria Hussein, Catherine Villeneuve, Jes Lefcourt, George Wittemyer, Mara Elephant Project, Allen Institute for Artificial Intelligence, Google Earth Outreach, Save the Elephants, Elephants Alive, Berkeley Data Science Society.
