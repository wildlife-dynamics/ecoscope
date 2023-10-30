.. image:: https://ecoscope.io/en/latest/_static/logo.svg
   :width: 400
   :height: 200
   :align: center
   :target: https://ecoscope.io

|PyPI| |Tests| |Codecov| |Docs| |Notebooks|

.. |PyPI| image:: https://img.shields.io/pypi/v/ecoscope.svg
   :target: https://pypi.python.org/pypi/ecoscope

.. |Tests| image:: https://github.com/wildlife-dynamics/ecoscope/actions/workflows/main.yml/badge.svg
   :target: https://github.com/wildlife-dynamics/ecoscope/actions?query=workflow%3ATests

.. |Codecov| image:: https://codecov.io/gh/wildlife-dynamics/ecoscope/branch/master/graphs/badge.svg
   :target: https://codecov.io/gh/wildlife-dynamics/ecoscope
   
.. |Docs| image:: https://readthedocs.org/projects/ecoscope/badge/?version=latest
   :target: https://ecoscope.io/en/latest/index.html

.. |Notebooks| image:: https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter
   :target: https://ecoscope.io/en/latest/notebooks.html

========
Ecoscope
========

Ecoscope is an open-source analysis module for tracking, environmental and conservation data analyses. It provides methods and approaches for: Data I/O (EarthRanger, Google Earth Engine, Landscape Dynamics, MoveBank, Geopandas), Movement Data (Relocations, Trajectories, Home-Ranges, EcoGraph, Recurse, Geofencing, Immobility, Moving Windows, Resampling, Filtering), Visualization (EcoMap, EcoPlot), Environmental analyses (Seasons determination, Remote sensing anomalies), Covariate labeling (Day/Night, Seasonal, GEE Image Collections/Images).

Development & Testing
=====================
Development dependencies are included in `environment.yml`. You will also need to configure the selenium driver following these instructions (note that this is not needed if running ecoscope within Google Colab as selenium is already configured): https://www.selenium.dev/documentation/webdriver/getting_started/install_drivers/

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
We thank the many individuals and organizations who have contributed to the development of Ecoscope: Jake Wall, Peter Kulits, Zakaria Hussein, Catherine Villeneuve, Jes Lefcourt, Chris Jones, George Wittemyer, Mara Elephant Project, Allen Institute for Artificial Intelligence, Google Earth Outreach, Save the Elephants, Elephants Alive, Berkeley Data Science Society.
