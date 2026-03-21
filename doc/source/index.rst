Ecoscope: Conservation Data Analytics
======================================

.. toctree::
   :titlesonly:
   :maxdepth: 2
   :hidden:

   Home <self>
   ecoscope_gui
   notebooks
   API Reference <autoapi/ecoscope/index>

Welcome to **Ecoscope**, an open-source Python platform for conservation data analytics and spatial analysis.
Built by `Wildlife Dynamics <https://www.wildlifedynamics.org/>`_ and the `Allen Institute for AI <https://allenai.org/>`_,
Ecoscope empowers wildlife researchers, conservationists, and data scientists to analyze movement data,
understand habitat use, and support data-driven conservation decisions.

---

🚀 **Quick Start**
==================

Get started with Ecoscope in minutes:

.. code-block:: python

   import ecoscope
   
   # Load and analyze wildlife movement data
   relocations = ecoscope.io.EarthRangerIO.get_relocations()
   
   # Analyze home range and movement patterns
   home_range = ecoscope.analysis.HomeRange(relocations)

👉 `Explore the Jupyter Notebooks <notebooks.html>`_ for hands-on examples.

---

✨ **Key Features**
===================

- **Data I/O**: Connect to EarthRanger, load GPS relocations and wildlife events
- **Trajectory Analysis**: Calculate movement metrics, speed, distance, turning angles
- **Home Range & Movescape**: Kernel density estimation, minimum convex polygons, movescape visualizations
- **Environmental Analysis**: Extract satellite data, NDVI, precipitation, elevation
- **Visualization**: Interactive maps with Deck.gl, static plots with Ecoscope GUI
- **Spatial Analysis**: Advanced geopandas-based spatial operations

---

📚 **Documentation**
====================

- **`Notebooks <notebooks.html>`_** - Interactive tutorials and examples
- **`Ecoscope GUI <ecoscope_gui.html>`_** - Web-based analysis interface
- **`API Reference <autoapi/ecoscope/index.html>`_** - Complete Python API documentation

---

🤝 **Community**
================

Ecoscope is built for and by the conservation research community.

- **GitHub**: `wildlife-dynamics/ecoscope <https://github.com/wildlife-dynamics/ecoscope>`_
- **Issues & Ideas**: `GitHub Discussions <https://github.com/wildlife-dynamics/ecoscope/discussions>`_
- **Contributing**: Check out `CONTRIBUTING.md <https://github.com/wildlife-dynamics/ecoscope/blob/main/CONTRIBUTING.md>`_

---

📖 **README**
=============

.. include:: ../../README.rst
