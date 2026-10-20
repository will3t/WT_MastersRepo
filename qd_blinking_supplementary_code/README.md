# QD blinking analysis companion code

This bundle supports the measurement/analysis workflow in *A practical guide for wide-field imaging of quantum dot blinking*. The notebook is the entry point; keep `qd_blinking.py` beside it. The numerical functions are factored into that readable module so they can be tested without loading experimental data.

## Start

Use Python 3.12. In a dedicated environment:

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name qd-blinking --display-name "QD blinking"
```

Open `paper_code.ipynb` in an existing Jupyter or VS Code notebook interface and choose this kernel. If a browser interface is needed, install JupyterLab (`python -m pip install jupyterlab`) and launch `jupyter lab`. Restart the kernel and run all cells in order.

The default `DEMO_MODE=True` creates two synthetic blinking spots, then runs detection, fitting, extraction, plotting and export. These are demonstration data, not an experimental result or a physical blinking-statistics model. The random seed and synthetic ground truth are recorded.

## Use a real recording

1. Set `DEMO_MODE=False` and leave `RAW_FILE=None`. The load cell opens a local file picker when available; otherwise it prompts for a path. No experimental filename or absolute input path is embedded in the release notebook. Cancelling stops explicitly. On remote Jupyter, the file must be accessible to the kernel; the picker is not a browser upload.
2. Verify width, height, byte order, header length, unpacked storage format and expected frame count from acquisition metadata. The original notebook used 1024 × 1024; the draft SI says 1058 × 1058. Do not guess. The loader checks bytes/count but cannot prove the geometry is right.
3. Specify exposure separately from frame interval. Set `USE_TIMESTAMP_FILE=True` to choose frame-start timestamps (CSV column `time_s`), or supply a verified regular interval. Seconds stay blank if timing is unknown; plots then use frame index. Irregular timestamps are used for plots but not converted into simple dwell durations.
4. Inspect the projection, candidate table and individual regions. Use IDs from this run; they are original detection IDs and are never renumbered after filtering. IDs from the old renumbered notebook must be matched by coordinates, not assumed to identify the same feature.
5. Adjust detection/fit settings only with a recorded rationale. Edge, bound and duplicate-centre rejections are additional quality safeguards and can change the candidate count compared with the original notebook.
6. Threshold analysis is disabled by default. If wanted, enter candidate-specific `THRESHOLDS` in corrected mean-pixel counts, after looking at the histograms. Above/below-threshold labels are descriptive, not validated ON/OFF assignments.

## Measurement definitions

The fitted Gaussian is `A exp(-((x-x0)^2+(y-y0)^2)/(2 R^2)) + B`; **R = sigma** in pixels. LoG detection sigma and its display radius sqrt(2)*sigma are distinct conventions.

Signal: mean pixel value inside radius 3. Background: median within radii 5–8. Pixel-centre boundaries are included. Corrected signal: signal minus background. Integrated corrected counts: corrected mean times the actual number of aperture pixels. This number depends on the fitted subpixel position and need not be 29. No detector gain, noise calibration or total-emitter encircled-energy correction is applied. Units are camera counts (ADU), not electrons or photons.

Trace standard deviation includes blinking; it is not an SNR estimate. A raw baseline can include camera bias, so the former signal/background ratio is omitted. The median background estimator does not make neighbouring emitters or spatially varying backgrounds harmless. Fixed apertures assume stable position/focus; this code does not track drift.

Optional threshold-run tables mark the first and last runs as censored. Durations in seconds are sampled run length times a verified regular frame interval, not directly measured transition times. They should not be used as an unbiased lifetime or power-law fit. No mean dwell-time estimator is supplied.

## Outputs

Each run creates a new directory under `qd_blinking_results/`:

- `all_candidates.csv`: every detection, fitted values, fit diagnostics and retention/rejection reason.
- `retained_candidates.csv`: accepted spatial candidates; not certified single QDs.
- `trace_statistics.csv`: descriptive statistics, actual mask counts and optional saturation fraction.
- `extracted_traces.csv`: full raw signal, background and corrected mean/integrated trajectories; same IDs as plots.
- `exploratory_threshold_runs.csv` and `exploratory_threshold_summary.csv`: empty, header-only tables when thresholds are disabled.
- `analysis_parameters.json`: input identity/hash, dimensions, time metadata, settings, software versions, numerical-module hash and demo flag.
- `candidate_*.png` / `.svg`: selected candidate figures, if enabled.

Mask counts and unrounded coordinates allow the extraction to be checked. `saturation_adu=None` means saturation was not assessed; it is not a zero-saturation result. SHA-256 hashing reads a large input file once and may take time. Existing result directories are never overwritten.

## Verification and limits

Run `python -m unittest -v test_qd_blinking.py`. Tests cover file selection and fallback, byte-order/header validation, exact extraction against an independent full-frame-mask reference, Gaussian recovery, timing, stable IDs, empty inputs/results, boundary censoring and synthetic end-to-end export.

The review tested the supplied qdb_20260824_102327.raw recording: 1000 frames of 1024 × 1024 uint16 pixels reproduced the original notebook results; revised extraction retained 168 candidates after removing one near-duplicate fit. Frame timing still needs acquisition metadata. Before submission, run this bundle on the other released videos and provide a figure manifest: panel, video filename/hash, candidate ID and position, frame range, timing, any manually chosen level/threshold and corresponding exported trace. Do not present synthetic demonstration output as a reconstruction of Figure 9.

## Citation, authorship and licence

Cite the accompanying methods paper; add its final bibliographic details/DOI when available. Method context is given in the notebook. Confirm the software author/rights-holder names and an explicit release licence with the responsible rights holders. This review does not assign a licence or determine ownership. Dependencies retain their respective licences. The separate review and IP notes are not part of this release bundle.

## Keep the distributed notebook small

The release notebook has saved cell outputs and execution counts cleared; it contains no embedded experimental movie or plot images. After running it, use your notebook editor’s **Clear All Outputs** command and save before distributing the source again. Keep an executed copy separately if desired. Figures and numerical results are exported to their own run directory. The original 48.0 MB notebook contained about 47.8 MB of encoded PNG outputs; the revised clean notebook is about 20 KB.
