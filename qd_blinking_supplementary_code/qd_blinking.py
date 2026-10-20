"""Candidate detection and trace extraction for the QD blinking methods paper.

R is the fitted circular Gaussian standard deviation, in pixels. This module
extracts candidate-emitter traces; it does not establish emitter number or
perform validated ON/OFF classification. See README and the notebook.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from skimage.feature import blob_log


@dataclass(frozen=True)
class AnalysisConfig:
    blob_min_sigma: float = 2.0
    blob_max_sigma: float = 6.0
    blob_num_sigma: int = 10
    blob_threshold: float = 0.05
    blob_overlap: float = 0.5
    fit_window_radius: int = 6
    fit_max_evaluations: int = 5000
    min_R: float = 1.5
    max_R: float = 6.0
    aperture_radius: float = 3.0
    background_inner: float = 5.0
    background_outer: float = 8.0
    chunk_frames: int = 64
    duplicate_distance: float = 1.0
    saturation_adu: float | None = None

    def validate(self):
        vals = [self.blob_min_sigma, self.blob_max_sigma, self.blob_threshold,
                self.blob_overlap, self.min_R, self.max_R, self.aperture_radius,
                self.background_inner, self.background_outer, self.duplicate_distance]
        if not np.all(np.isfinite(vals)):
            raise ValueError('Analysis parameters must be finite.')
        if not 0 < self.blob_min_sigma <= self.blob_max_sigma:
            raise ValueError('Require 0 < blob_min_sigma <= blob_max_sigma.')
        if self.blob_threshold <= 0 or not 0 <= self.blob_overlap <= 1:
            raise ValueError('Invalid detection threshold or overlap.')
        if not 0 < self.min_R < self.max_R:
            raise ValueError('Require 0 < min_R < max_R.')
        if not 0 < self.aperture_radius < self.background_inner < self.background_outer:
            raise ValueError('Require 0 < aperture < background inner < background outer.')
        for name in ['blob_num_sigma', 'fit_window_radius', 'fit_max_evaluations', 'chunk_frames']:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.duplicate_distance < 0:
            raise ValueError('duplicate_distance must be nonnegative.')
        if self.saturation_adu is not None and (not np.isfinite(self.saturation_adu) or self.saturation_adu <= 0):
            raise ValueError('saturation_adu must be positive or None.')



def select_input_file(description="RAW microscopy video", file_pattern="*.raw"):
    """Ask for an existing file; no machine-specific filename is stored in code.

    Uses a local Tk file picker when available; otherwise prompts in the notebook.
    On a remote kernel, the selected/path-entered file must be visible to that
    kernel. Large RAW movies are not uploaded into notebook/browser memory.
    Cancelling is explicit and does not silently select a different recording.
    """
    selected = None
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        tk = None
    if tk is not None:
        try:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(
                parent=root, title=f"Select {description}",
                filetypes=[(description, file_pattern), ("All files", "*")])
        except tk.TclError:
            selected = None
        finally:
            if root is not None:
                root.destroy()
    if selected is None:
        selected = input(f"Path to {description} (visible to this Python kernel): ").strip()
    if not selected:
        raise FileNotFoundError(f"No {description} selected. Rerun this cell to choose a file.")
    # Accept a quoted path pasted from a file manager; spaces inside are preserved.
    if len(selected) > 1 and selected[0] == selected[-1] and selected[0] in ("'", '"'):
        selected = selected[1:-1]
    path = Path(selected).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Selected file does not exist: {path}")
    return path.resolve()


def load_raw_video(filename, width, height, header_bytes=0, dtype='<u2', expected_frames=None):
    """Memory-map a headerless/known-header sequence of unpacked uint16 frames.

    No automatic geometry inference: several wrong shapes can divide a file
    exactly. Byte order, dimensions, padding and header must come from acquisition
    metadata. Packed 12-bit streams, per-frame headers and row padding unsupported.
    """
    for name, value in [('width', width), ('height', height), ('header_bytes', header_bytes)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer.')
    if width <= 0 or height <= 0 or header_bytes < 0:
        raise ValueError('Dimensions must be positive and header_bytes nonnegative.')
    if dtype not in ('<u2', '>u2'):
        raise ValueError("Specify explicit uint16 byte order: '<u2' or '>u2'.")
    if expected_frames is not None and (isinstance(expected_frames, bool) or
            not isinstance(expected_frames, (int, np.integer)) or expected_frames < 1):
        raise ValueError('expected_frames must be a positive integer or None.')
    p = Path(filename)
    payload = p.stat().st_size - header_bytes
    frame_bytes = width * height * 2
    if payload <= 0 or payload % frame_bytes:
        raise ValueError(f'{payload} payload bytes do not form complete {width} x {height} uint16 frames.')
    count = payload // frame_bytes
    if expected_frames is not None and count != expected_frames:
        raise ValueError(f'Found {count} frames; expected {expected_frames}. Check RAW metadata.')
    return np.memmap(p, dtype=dtype, mode='r', offset=header_bytes, shape=(count, height, width))


def timing(n_frames, exposure_s, frame_interval_s=None, timestamps_s=None):
    """Return elapsed frame-start times and a verified regular interval, if known.

    Exposure is not inferred to be the frame interval. With unknown timing,
    time_s is NaN and plots use frame indices. Irregular timestamps are retained,
    but frame-based dwell summaries in seconds are disabled.
    """
    if n_frames < 1 or not np.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError('Need at least one frame and a positive finite exposure.')
    if frame_interval_s is not None:
        if not np.isfinite(frame_interval_s) or frame_interval_s < exposure_s:
            raise ValueError('Frame interval must be finite and at least the exposure time.')
    regular = frame_interval_s
    if timestamps_s is not None:
        t = np.asarray(timestamps_s, dtype=float)
        if t.shape != (n_frames,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Provide one finite, strictly increasing frame-start timestamp per frame.')
        delta = np.diff(t)
        if len(delta) and np.any(delta < exposure_s - max(1e-9, exposure_s * 1e-6)):
            raise ValueError('Timestamp intervals shorter than exposure: check units/metadata.')
        if frame_interval_s is not None and len(delta) and not np.allclose(delta, frame_interval_s, rtol=1e-5, atol=1e-9):
            raise ValueError('Timestamps disagree with the specified frame interval.')
        regular = float(np.median(delta)) if len(delta) and np.allclose(delta, np.median(delta), rtol=1e-5, atol=1e-9) else None
        t = t - t[0]
    elif frame_interval_s is not None:
        t = np.arange(n_frames, dtype=float) * frame_interval_s
    else:
        t = np.full(n_frames, np.nan)
    return t, regular


def projections(video, chunk_frames=64):
    if video.ndim != 3 or min(video.shape) < 1 or chunk_frames < 1:
        raise ValueError('Expected a non-empty (frames, height, width) video and positive chunk size.')
    total = np.zeros(video.shape[1:], dtype=np.float64)
    maximum = np.full(video.shape[1:], -np.inf)
    for start in range(0, len(video), chunk_frames):
        chunk = video[start:start + chunk_frames]
        if not np.isfinite(chunk).all():
            raise ValueError('Video contains non-finite pixel values.')
        total += np.sum(chunk, axis=0, dtype=np.float64)
        np.maximum(maximum, np.max(chunk, axis=0), out=maximum)
    return total / len(video), maximum


def gaussian_2d(coords, amplitude, x0, y0, sigma, background):
    x, y = coords
    return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)) + background


FIT_COLUMNS = ['candidate_id', 'detected_x', 'detected_y', 'detected_sigma',
               'x', 'y', 'R', 'amplitude', 'fit_background', 'fit_rmse',
               'fit_at_bound', 'fit_covariance_finite', 'fit_ok', 'retained', 'reason']


def fit_candidates(image, cfg=AnalysisConfig()):
    cfg.validate()
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError('Need a finite non-empty 2D image.')
    span = np.ptp(image)
    if span <= 0:
        return pd.DataFrame(columns=FIT_COLUMNS)
    normalised = (image-image.min()) / span
    blobs = blob_log(normalised, min_sigma=cfg.blob_min_sigma,
                     max_sigma=cfg.blob_max_sigma, num_sigma=cfg.blob_num_sigma,
                     threshold=cfg.blob_threshold, overlap=cfg.blob_overlap)
    rows = []
    for cid, (y, x, guess) in enumerate(blobs, 1):
        row = dict.fromkeys(FIT_COLUMNS, np.nan)
        row.update(candidate_id=cid, detected_x=x, detected_y=y, detected_sigma=guess,
                   fit_ok=False, retained=False, reason='fit_failed')
        cx, cy = int(round(x)), int(round(y))
        r = cfg.fit_window_radius
        if cx-r < 0 or cy-r < 0 or cx+r >= image.shape[1] or cy+r >= image.shape[0]:
            row['reason'] = 'fit_window_at_edge'; rows.append(row); continue
        patch = image[cy-r:cy+r+1, cx-r:cx+r+1]
        yy, xx = np.indices(patch.shape)
        bg = float(np.median(patch))
        lower = np.array([0, 0, 0, 0.5, -np.inf])
        upper = np.array([np.inf, 2*r, 2*r, 20, np.inf])
        p0 = [max(float(patch.max()-bg), 1e-6), r, r, np.clip(guess, 0.51, 19.99), bg]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                p, cov = curve_fit(gaussian_2d, (xx.ravel(), yy.ravel()), patch.ravel(),
                                   p0=p0, bounds=(lower, upper), maxfev=cfg.fit_max_evaluations)
            if not np.isfinite(p).all():
                raise ValueError('Non-finite fit')
        except (RuntimeError, ValueError, FloatingPointError):
            rows.append(row); continue
        a, xf, yf, sigma, b = p
        bound = bool(np.any(np.isclose(p, lower, atol=1e-4, rtol=0)) or
                     np.any(np.isclose(p, upper, atol=1e-4, rtol=0)))
        xf, yf = xf+cx-r, yf+cy-r
        row.update(x=xf, y=yf, R=sigma, amplitude=a, fit_background=b,
                   fit_rmse=float(np.sqrt(np.mean((gaussian_2d((xx, yy), *p)-patch)**2))),
                   fit_at_bound=bound, fit_covariance_finite=bool(np.isfinite(cov).all()), fit_ok=True)
        reason = 'retained'
        if not cfg.min_R <= sigma < cfg.max_R: reason = 'width_outside_range'
        elif bound: reason = 'fit_at_bound'
        elif not full_annulus(image.shape, xf, yf, cfg.background_outer): reason = 'annulus_at_edge'
        elif any(q['retained'] and np.hypot(xf-q['x'], yf-q['y']) < cfg.duplicate_distance for q in rows):
            reason = 'duplicate_fitted_centre'
        row.update(retained=(reason == 'retained'), reason=reason)
        rows.append(row)
    # IDs identify original detections; filtering never renumbers them.
    return pd.DataFrame(rows, columns=FIT_COLUMNS)


def full_annulus(shape, x, y, radius):
    return bool(np.isfinite([x, y, radius]).all() and radius > 0 and
                x-radius >= 0 and y-radius >= 0 and
                x+radius <= shape[1]-1 and y+radius <= shape[0]-1)


def local_masks(shape, x, y, cfg=AnalysisConfig()):
    cfg.validate()
    if not full_annulus(shape, x, y, cfg.background_outer):
        raise ValueError('Full background annulus must fit inside the image.')
    r = cfg.background_outer
    x0, x1 = max(0, int(np.floor(x-r))), min(shape[1], int(np.ceil(x+r))+1)
    y0, y1 = max(0, int(np.floor(y-r))), min(shape[0], int(np.ceil(y+r))+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx-x)**2 + (yy-y)**2
    signal = d2 <= cfg.aperture_radius**2
    background = (d2 >= cfg.background_inner**2) & (d2 <= cfg.background_outer**2)
    if not signal.any() or not background.any():
        raise ValueError('Empty aperture or background mask.')
    return (slice(y0, y1), slice(x0, x1)), signal, background


def extract_trace(video, x, y, cfg=AnalysisConfig()):
    region, mask, annulus = local_masks(video.shape[1:], x, y, cfg)
    n = len(video)
    if n < 1: raise ValueError('Video is empty.')
    signal, background = np.empty(n), np.empty(n)
    saturated = 0
    for start in range(0, n, cfg.chunk_frames):
        end = min(n, start+cfg.chunk_frames)
        chunk = np.asarray(video[start:end, region[0], region[1]], dtype=float)
        if not np.isfinite(chunk).all(): raise ValueError('Non-finite pixels in extraction region.')
        values = chunk[:, mask]
        signal[start:end] = values.mean(axis=1)
        background[start:end] = np.median(chunk[:, annulus], axis=1)
        if cfg.saturation_adu is not None:
            saturated += int(np.count_nonzero(values >= cfg.saturation_adu))
    corrected = signal-background
    return dict(signal=signal, background=background, corrected=corrected,
                integrated_corrected=corrected*int(mask.sum()),
                aperture_pixels=int(mask.sum()), annulus_pixels=int(annulus.sum()),
                saturated_aperture_fraction=(saturated/(n*mask.sum()) if cfg.saturation_adu is not None else np.nan))


STAT_COLUMNS = ['candidate_id', 'mean_corrected_counts', 'median_corrected_counts',
                'std_corrected_counts', 'min_corrected_counts', 'max_corrected_counts',
                'mean_background_counts', 'aperture_pixels', 'annulus_pixels',
                'saturated_aperture_fraction']


def trace_statistics(traces):
    rows = []
    for cid, tr in traces.items():
        c = tr['corrected']
        rows.append(dict(candidate_id=cid, mean_corrected_counts=np.mean(c),
            median_corrected_counts=np.median(c), std_corrected_counts=np.std(c, ddof=0),
            min_corrected_counts=np.min(c), max_corrected_counts=np.max(c),
            mean_background_counts=np.mean(tr['background']), aperture_pixels=tr['aperture_pixels'],
            annulus_pixels=tr['annulus_pixels'], saturated_aperture_fraction=tr['saturated_aperture_fraction']))
    return pd.DataFrame(rows, columns=STAT_COLUMNS)


DWELL_COLUMNS = ['state', 'start_frame', 'stop_frame_exclusive', 'n_frames',
                 'left_censored', 'right_censored', 'complete', 'sampled_duration_s']


def threshold_runs(corrected, threshold, frame_interval_s=None):
    """Exploratory threshold runs, not physical ON/OFF or lifetime estimates.

    The threshold must be chosen in background-corrected mean-pixel counts.
    First and last runs are censored. Seconds use n_frames * regular interval;
    transition boundaries are only resolved to acquisition bins. No power-law fit.
    """
    c = np.asarray(corrected, dtype=float)
    if c.ndim != 1 or not c.size or not np.isfinite(c).all() or not np.isfinite(threshold):
        raise ValueError('Need a finite non-empty trace and finite corrected-count threshold.')
    if frame_interval_s is not None and (not np.isfinite(frame_interval_s) or frame_interval_s <= 0):
        raise ValueError('Interval must be positive or None.')
    state = c > threshold
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1])+1]
    ends = np.r_[starts[1:], len(c)]
    rows = []
    for start, end in zip(starts, ends):
        rows.append(dict(state='above' if state[start] else 'below_or_equal',
                         start_frame=int(start), stop_frame_exclusive=int(end), n_frames=int(end-start),
                         left_censored=bool(start == 0), right_censored=bool(end == len(c)),
                         complete=bool(start > 0 and end < len(c)),
                         sampled_duration_s=(float((end-start)*frame_interval_s) if frame_interval_s is not None else np.nan)))
    return pd.DataFrame(rows, columns=DWELL_COLUMNS)


def synthetic_video(seed=20261006, n_frames=120):
    """Small deterministic demonstration. NOT experimental data or an InP model."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:80, :80]
    video = np.full((n_frames, 80, 80), 200.0)
    truth = []
    for x, y, sigma, period in [(24.3, 26.6, 1.8, 15), (55.2, 52.1, 2.1, 23)]:
        on = (np.arange(n_frames)//period) % 2 == 1
        amplitude = 130*on
        g = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))
        video += amplitude[:, None, None]*g
        truth.append(dict(x=x, y=y, sigma=sigma, bright_frames=on.tolist()))
    video += rng.normal(0, 3, video.shape)
    return np.rint(np.clip(video, 0, 65535)).astype('<u2'), truth


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def export_results(directory, fits, traces, time_s, metadata, cfg, thresholds=None, regular_interval=None):
    """Create a NEW run directory; never mix with an earlier run's outputs."""
    p = Path(directory)
    if p.exists(): raise FileExistsError(f'Use a new run directory: {p}')
    n = len(time_s)
    if any(len(tr['corrected']) != n for tr in traces.values()):
        raise ValueError('Time axis and traces differ in length.')
    thresholds = {} if thresholds is None else thresholds
    if set(thresholds)-set(traces): raise ValueError('Threshold refers to a missing candidate ID.')
    if any(not np.isfinite(v) for v in thresholds.values()): raise ValueError('Thresholds must be finite.')
    p.mkdir(parents=True)
    fits.to_csv(p/'all_candidates.csv', index=False)
    fits.loc[fits['retained'].astype(bool)].to_csv(p/'retained_candidates.csv', index=False)
    stats = trace_statistics(traces)
    stats.to_csv(p/'trace_statistics.csv', index=False)
    trace_columns = ['candidate_id', 'frame', 'time_s', 'signal_mean_counts', 'background_median_counts',
                     'corrected_mean_counts', 'corrected_integrated_counts', 'aperture_pixels', 'annulus_pixels']
    with open(p/'extracted_traces.csv', 'w') as stream:
        pd.DataFrame(columns=trace_columns).to_csv(stream, index=False)
        for cid, tr in traces.items():
            pd.DataFrame(dict(candidate_id=cid, frame=np.arange(n), time_s=time_s,
                signal_mean_counts=tr['signal'], background_median_counts=tr['background'],
                corrected_mean_counts=tr['corrected'], corrected_integrated_counts=tr['integrated_corrected'],
                aperture_pixels=tr['aperture_pixels'], annulus_pixels=tr['annulus_pixels'])).to_csv(stream, header=False, index=False)
    all_runs, summaries = [], []
    for cid, th in thresholds.items():
        runs = threshold_runs(traces[cid]['corrected'], th, regular_interval)
        runs.insert(0, 'candidate_id', cid); runs.insert(1, 'threshold_corrected_counts', th)
        all_runs.append(runs)
        summaries.append(dict(candidate_id=cid, threshold_corrected_counts=th,
            fraction_frames_above=float(np.mean(traces[cid]['corrected'] > th)),
            transitions=len(runs)-1, complete_runs=int(runs['complete'].sum())))
    (pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame(columns=['candidate_id', 'threshold_corrected_counts']+DWELL_COLUMNS)).to_csv(p/'exploratory_threshold_runs.csv', index=False)
    pd.DataFrame(summaries, columns=['candidate_id', 'threshold_corrected_counts', 'fraction_frames_above', 'transitions', 'complete_runs']).to_csv(p/'exploratory_threshold_summary.csv', index=False)
    versions = {name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-image','matplotlib']}
    report = dict(metadata, analysis_config=asdict(cfg), thresholds_corrected_counts=thresholds,
                  python_version=platform.python_version(), package_versions=versions,
                  module_sha256=sha256_file(__file__), regular_frame_interval_s=regular_interval,
                  fitted_R_definition='circular Gaussian sigma in pixels',
                  timing_known=bool(n and np.isfinite(time_s).all()),
                  trace_definition='aperture mean minus local annular median; ADU counts per pixel',
                  code_version='2026-10-06-reviewed')
    with open(p/'analysis_parameters.json', 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return p
