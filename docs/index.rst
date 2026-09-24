Winery Adventures
=================

Winery Adventures is a data pipeline that analyzes the sensor readings (pH,
temperature and volume of must) of the fermentation tanks of a small winery.
It adds derived columns to the readings and computes a *fermentation stress*
score for every tank.

Quick start
-----------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate winery
   python -m winery_adventures.main --input data/sensors_sample.tsv \
       --tank-info data/tank_info_sample.tsv --output results.csv

The complete instructions (tests, benchmark, code style) are in the
``README.md`` file of the repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api
   design
