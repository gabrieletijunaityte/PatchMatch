# Matching Marine Debris Patches Across PlanetScope and Sentinel-2 Double Acquisitions (MSc Thesis Project)

---

## About the project
This repository contains the dataset, code and experimental model setups for my master's thesis on matching marine debris (MD) patches across multi-platform and multi-temporal imagery (PlanetScope and Sentinel-2 double acquisitions acquired within a one-hour interval). This project involves dataset collection, the Double Acquisition NN development through iterative experimental process and experiments on applying different level drift knowledge to simulate what kind of model performance can be expected in applied scenarios.

## Repository Structure

<pre lang="md">
<code>.
├── config/ # Contains all experimental model configuration files
│   ├── *.json # Model configuration settings
│   ├── configs.py # A class to load in configuration file and initialise some parts of the models.
│   └── summarise.py # Script to summarise all configuration files into a .csv.
├── data/
│   ├── annotations/ # .geojson annotations files organised per study events.
│   ├── PS/ # Folder for all PS scenes organised per study events 
│   ├── S2/ # Folder for all S2 scenes organised per study events 
│   ├── styles/ # QGIS style files for annotation process
│   ├── test_areas/ # Folder with polygons for test areas
│   ├── validate_areas/ # Folder with polygons for validation areas
│   └── database.db # Databse of the annotations and scenes
├── data_preparation/
│   ├── annotations/
│   │   ├── annotation_data.py # Deal with annotation files (.geojson)
│   │   ├── create_qgis_projects.py # Creates QGIS projects for annotations
│   │   └── get_ndi.py # Prepares NDI rasters for scenes
│   ├── data_inspection.py # Data inspection functions (no data, histograms)
│   ├── obtain_S2.py # Downloads data from google earth engine
│   ├── prepare_PS.py # Converts PS data to TOA-reflectance
│   └── tile_processing.py # Extracts tiles from scenes
├── database/
│   ├── db_quaries.py # SQL code queries that are used in database/ProjectDB.py
│   └── ProjectDB.py # Python class for project database (sql)
├── intermediate/
│   └── tiles/
│       ├── *.tif # Extracted (with slightly larger extent, see pre-tiles in the report).
│       ├── ssl_365.json # Map of train/validate/test split per platform.
│       ├── test_tiles_365.json # Map for positive test pairs 
|       └── tiles_365.json # Map for positive all pairs
├── output/
│   ├── figures/
│   ├── predictions/ # Prediction matrices
│   ├── results/ # Logged test results
│   └── weights/ # Trained model weigths
├── plotting/ # functions to create illustrations for the report
├── testing/
│   ├── classification_metrics.py # Calculates classification metrics
│   └── retrieval_metrics.py # Calculates retrieval metrics
├── training/
│   ├── datamodule/ # Datamodules and dataloaders
│   ├── models/ # Architectures
│   │   ├── backbones.py 
│   │   ├── losses.py
│   │   ├── projection_heads.py
│   │   ├── PS_encoders.py
│   │   ├── S2_encoders.py
|   │   └── similarities.py
│   ├── proposed_model.py # Model implementation
│   └── ssl_pre_training.py # Model pre-training implementation
├── utils/ # Extras
├── weights/ # Original SSL4EO weigths
├── main_*.py # Main scripts to run scripts.
├── README.md
└── requirements.txt </code>
</pre>


## Getting Started

### Cloning
<pre lang="bash">
git clone https://github.com/gabrieletijunaityte/DoubleAcquisitionNN.git
cd DoubleAcquisitionNN
</pre>

### Environment
<pre lang="bash">
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
</pre>

### Running the experiments and main model
Note: experiment choice can be provided by selecting a different configuration file.

For data acquisition:
<pre lang="bash">
python3 main_data_acquisition.py config/proposed.json
</pre>

For data annotation process and/or annotation extraction to database:
<pre lang="bash">
python3 main_data_annotations.py config/proposed.json
</pre>

For tile generation from the database
<pre lang="bash">
python3 main_data_preparation.py config/proposed.json
</pre>

For SSL pretraining:
<pre lang="bash">
python3 main_ssl_training.py config/refined_best.json
</pre>

For model training on marine debris patch matching task:
<pre lang="bash">
python3 main_training.py config/proposed.json
</pre>

For trained model testing:
<pre lang="bash">
python3 main_training.py config/proposed.json
</pre>


## Dataset
Raw imagery is not included in the git repository due to licensing constraints.