# Supporting Material for Thesis

This repository accompanies the thesis:

**Impact of Ti:sapphire Laser Irradiation in Continuous Wave and Femtosecond Pulsed Modes, Including Second Harmonic Generated Output, on *Ascaris suum* Eggs: An In Ovo Analysis of Biological Effects Relevant to Attenuation**

The repository is intended as supporting material for the experimental workflow, raw and derived data organization, computer-vision-assisted irradiation pipeline, and downstream statistical analysis used in the study.

Repository DOI (Zenodo): [10.5281/zenodo.19020088](https://doi.org/10.5281/zenodo.19020088)

## Overview

The repository contains four main components:

1. `exp_data/vid_embryoscope/`: compressed time-lapse embryoscope videos used as the primary raw observation material.
2. `exp_data/data_analysis/`: master tables, Jupyter notebooks, and derived plots/tables used for manual analysis and statistical summaries.
3. `lia-control/`: the laboratory automation scripts used to detect eggs in live camera images and irradiate them automatically using a motorized stage and shutter.
4. `cv_model/`: the trained YOLO-based egg detection model and related training/validation artifacts.

The video material in `exp_data/vid_embryoscope/` is highly compressed for repository distribution. Higher-quality source videos can be provided on request.

## Repository Layout

```text
.
├── cv_model/
│   └── egg-detector_run_CURTA_A5000_b32_iz800_e200/
├── exp_data/
│   ├── data_analysis/
│   │   ├── 254nm/
│   │   ├── 400nm/
│   │   ├── 800nm/
│   │   ├── cross-modality/
│   │   ├── master_csv/
│   │   └── analysis_*.ipynb
│   └── vid_embryoscope/
│       ├── controls/
│       ├── 254nm/
│       ├── 400nm/
│       └── 800nm/
├── inference_egg-detector.py
├── lia-control/
│   ├── lia_auto_irradiation.py
│   ├── stage_dll_loader.py
│   └── libximc_2.13.2/
└── requirements.txt
```

## Experimental Data

### `exp_data/vid_embryoscope/`

This folder contains the time-lapse videos used for biological assessment of irradiated and control eggs. The videos are grouped by irradiation modality:

- `controls/`
- `254nm/`
- `400nm/`
- `800nm/`

Within each modality, the material is further separated into:

- `embryonated/`
- `unembryonated/`

These videos represent the raw observational data used for manual annotation and biological outcome assessment.

### `exp_data/data_analysis/master_csv/`

This folder contains the master analysis tables generated from manual review of the embryoscope videos:

- `01_master_unembryonated.csv`
- `02_master_embryonated.csv`

These master CSV files are the central tabular inputs for the Jupyter notebook-based analysis workflow.

### `exp_data/data_analysis/`

This folder contains the analysis notebooks and the derived output tables/plots used in the thesis.

Included notebooks:

- `analysis_254_embryonated.ipynb`
- `analysis_254_unembryonated.ipynb`
- `analysis_400_embryonated.ipynb`
- `analysis_400_unembryonated.ipynb`
- `analysis_800_embryonated.ipynb`
- `analysis_800_unembryonated.ipynb`
- `analysis_controls_embryonated.ipynb`
- `analysis_sumup_embryonated.ipynb`
- `analysis_sumup_unembryonated.ipynb`

The analysis outputs are organized by modality and developmental stage, for example:

- `254nm/embryonated/plots` and `254nm/embryonated/tables`
- `254nm/unembryonated/plots` and `254nm/unembryonated/tables`
- `400nm/embryonated/plots` and `400nm/embryonated/tables`
- `400nm/unembryonated/plots` and `400nm/unembryonated/tables`
- `800nm/embryonated/plots` and `800nm/embryonated/tables`
- `800nm/unembryonated/plots` and `800nm/unembryonated/tables`
- `cross-modality/embryonated/plots` and `cross-modality/embryonated/tables`
- `cross-modality/unembryonated/plots` and `cross-modality/unembryonated/tables`

These outputs include summary tables, significance testing results, survival analyses, quality-control tables, and plotting data used in the thesis figures and interpretation.

## Computer Vision Model

The trained detection model is stored in:

- `cv_model/egg-detector_run_CURTA_A5000_b32_iz800_e200/weights/best.pt`

This model is used by the irradiation automation script to detect egg positions in live camera frames. The same folder also contains validation and training artifacts such as:

- `results.csv`
- training batch preview images
- validation curves and confusion matrices in `val_conf0.8/`

An additional helper script, `inference_egg-detector.py`, is included for batch inference on image folders using Ultralytics YOLO and SAHI-based sliced prediction.

Example:

```bash
python inference_egg-detector.py \
  --input_dir path/to/images \
  --model_path cv_model/egg-detector_run_CURTA_A5000_b32_iz800_e200/weights/best.pt
```

## Automated Irradiation Workflow

The main automation script is:

- `lia-control/lia_auto_irradiation.py`

This script was written for the original laboratory setup and performs the following high-level workflow:

1. Initializes the motorized stage through the bundled XIMC Python wrapper.
2. Connects to a Basler camera through `pypylon`.
3. Connects to an Arduino-controlled shutter through `pyfirmata`.
4. Displays a live camera stream for manual positioning.
5. Runs YOLO-based egg detection on the live frame.
6. Moves the stage to each detected target.
7. Performs a correction pass to refine centering.
8. Opens the shutter for the configured exposure time.
9. Logs shutter events and saves image snapshots during the run.

### Related files

- `lia-control/stage_dll_loader.py`: helper module for loading `pyximc`, opening the X/Y stage devices, and issuing movement commands.
- `lia-control/libximc_2.13.2/`: bundled third-party XIMC library distribution used by the stage loader.

### Current script assumptions

The script is tailored to the original lab hardware and should be treated as an experimental control script, not as a turnkey general-purpose package. In its current form it assumes:

- a Windows-based environment for stage control
- a Basler camera recognized by `pypylon`
- a Standa/XIMC-compatible two-axis stage
- an Arduino-connected shutter
- default Arduino settings `COM3` and digital pin `13`
- image output written to `C:/lab/MBI/img/<YYYY-MM-DD>`

Several acquisition parameters are configured near the top of `lia_auto_irradiation.py`, including:

- `STAGE_SPEED`
- `EXPOSURE_TIME_SEC`
- `STEP_SIZE`
- `YOLO_MODEL_PATH`
- `ARDUINO_PORT`
- `ARDUINO_PIN`

### Key bindings in `lia_auto_irradiation.py`

During live operation, the following keyboard shortcuts are used:

| Key | Function |
| --- | --- |
| `w` | Move stage up (negative Y direction) |
| `a` | Move stage left (negative X direction) |
| `s` | Move stage down (positive Y direction) |
| `d` | Move stage right (positive X direction) |
| `m` | Start the automated irradiation routine |
| `o` | Open the shutter manually |
| `c` | Close the shutter manually |
| `k` | Stop the program and close the shutter |
| `Esc` | Exit the live camera window |

The live preview window also displays:

- the last pressed key
- a total elapsed runtime counter after irradiation has started
- shutter status (`gate opened` / `gate closed`)
- a shutter timer while the gate is open

### Outputs produced by the automation script

The script generates or updates:

- date-specific image snapshots in `C:/lab/MBI/img/<YYYY-MM-DD>`
- shutter event logs in `lia-control/QC/<YYYY-MM-DD>.csv`

## Installation

Create a Python environment and install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

If you plan to run the hardware-control script `lia-control/lia_auto_irradiation.py`, a conservative full install command is:

```bash
python -m pip install -r requirements.txt pypylon pyfirmata pynput
```

For the automation workflow in `lia-control/`, a functioning hardware environment is also required:

- Basler pylon camera runtime/SDK
- Arduino accessible from the configured serial port
- XIMC stage libraries, which are bundled in this repository under `lia-control/libximc_2.13.2/`

## Running the Analysis Notebooks

To inspect or re-run the notebook workflow:

```bash
jupyter lab
```

Then open the notebooks in `exp_data/data_analysis/` and work from the relevant master tables in `exp_data/data_analysis/master_csv/`.

In general:

1. Use the master CSVs as the canonical manually curated input tables.
2. Run the modality-specific notebooks for embryonated or unembryonated eggs.
3. Use the `analysis_sumup_*.ipynb` notebooks for cross-modality comparisons and summary outputs.

## Notes and Limitations

- The embryoscope videos in this repository are compressed to make repository distribution practical.
- The automation code reflects the original laboratory hardware configuration and may require adaptation for another microscope, stage, camera, shutter controller, or filesystem layout.
- The script currently contains hard-coded assumptions for acquisition output directories and serial hardware settings.
- The bundled `libximc_2.13.2` folder is third-party software and includes its own upstream license file.

## License

This repository is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

It also uses **Ultralytics YOLOv11** through the `ultralytics` package and the trained model assets included in this repository. Ultralytics YOLO is offered under **AGPL-3.0** for open-source use, with a separate enterprise license available from Ultralytics for commercial use outside AGPL-3.0 terms.

## Citation

If you use this repository as supporting material, please cite the repository archive DOI and, where relevant, the associated thesis.

Repository archive DOI:

> 10.5281/zenodo.19020088

Persistent link:

> https://doi.org/10.5281/zenodo.19020088

Suggested thesis title for citation:

> Impact of Ti:sapphire Laser Irradiation in Continuous Wave and Femtosecond Pulsed Modes, Including Second Harmonic Generated Output, on *Ascaris suum* Eggs: An In Ovo Analysis of Biological Effects Relevant to Attenuation
