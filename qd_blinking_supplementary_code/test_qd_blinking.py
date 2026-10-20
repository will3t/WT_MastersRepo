"""Regression tests. Run: python -m unittest -v test_qd_blinking.py"""
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import ModuleType
from pathlib import Path
import numpy as np
import pandas as pd
from qd_blinking import (AnalysisConfig, gaussian_2d, load_raw_video, timing,
    projections, fit_candidates, local_masks, extract_trace, threshold_runs,
    synthetic_video, export_results, FIT_COLUMNS, select_input_file)

class FileSelectionTests(unittest.TestCase):
    def test_typed_path_with_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'recording with spaces.raw';p.touch()
            with patch.dict('sys.modules', {'tkinter': None}), patch('builtins.input', return_value=f'"{p}"'):
                self.assertEqual(select_input_file(),p.resolve())
    def test_typed_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict('sys.modules', {'tkinter': None}), patch('builtins.input', return_value=str(Path(d)/'missing.raw')):
                with self.assertRaises(FileNotFoundError):select_input_file()
    def test_picker_selection_and_cancellation(self):
        tk=ModuleType('tkinter');tk.TclError=RuntimeError
        window=Mock();tk.Tk=Mock(return_value=window);tk.filedialog=Mock()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'recording.raw';p.touch()
            tk.filedialog.askopenfilename.return_value=str(p)
            with patch.dict('sys.modules', {'tkinter':tk}):
                self.assertEqual(select_input_file(),p.resolve())
                window.destroy.assert_called_once()
                tk.filedialog.askopenfilename.return_value=''
                with self.assertRaises(FileNotFoundError):select_input_file()
    def test_no_display_falls_back_to_prompt(self):
        tk=ModuleType('tkinter');tk.TclError=RuntimeError
        tk.Tk=Mock(side_effect=RuntimeError('no display'));tk.filedialog=Mock()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'recording.raw';p.touch()
            with patch.dict('sys.modules', {'tkinter':tk}), patch('builtins.input',return_value=str(p)):
                self.assertEqual(select_input_file(),p.resolve())

class RawAndTimingTests(unittest.TestCase):
    def test_raw_header_and_endianness(self):
        with tempfile.TemporaryDirectory() as d:
            a = np.arange(24).reshape(2,3,4)*301
            for endian in ['<u2','>u2']:
                p=Path(d)/'input.raw';p.write_bytes(b'header'+a.astype(endian).tobytes())
                v=load_raw_video(p,4,3,6,endian,2)
                np.testing.assert_array_equal(v,a)
                self.assertFalse(v.flags.writeable)
    def test_raw_rejects_incomplete_bytes_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.raw'
            for data in [b'',b'x',b'\x00'*25]:
                p.write_bytes(data)
                with self.assertRaises(ValueError):load_raw_video(p,3,4)
    def test_raw_expected_count_and_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v.raw';p.write_bytes(bytes(48))
            for args in [dict(width=4,height=3,expected_frames=3),dict(width=0,height=3),
                         dict(width=4,height=3,header_bytes=None),dict(width=4,height=3,dtype='uint16')]:
                with self.assertRaises(ValueError):load_raw_video(p,**args)
    def test_exposure_not_frame_interval(self):
        t,dt=timing(3,0.05)
        self.assertTrue(np.isnan(t).all());self.assertIsNone(dt)
        t,dt=timing(3,0.05,0.08)
        np.testing.assert_allclose(t,[0,.08,.16]);self.assertEqual(dt,.08)
    def test_timestamp_validation_and_irregularity(self):
        t,dt=timing(3,.05,timestamps_s=[10,10.06,10.14])
        np.testing.assert_allclose(t,[0,.06,.14]);self.assertIsNone(dt)
        for stamps in [[0,0,.1],[0,.01,.1],[0,float('nan'),.1],[0,.1]]:
            with self.assertRaises(ValueError):timing(3,.05,timestamps_s=stamps)
        with self.assertRaises(ValueError):timing(3,.05,.05,[0,.06,.12])
    def test_invalid_frame_interval(self):
        for dt in [-1,0,.01,float('nan')]:
            with self.assertRaises(ValueError):timing(10,.05,dt)

class NumericalTests(unittest.TestCase):
    def test_projections_match_direct_operations(self):
        v,_=synthetic_video(n_frames=17)
        mean,maximum=projections(v,3)
        np.testing.assert_allclose(mean,v.mean(axis=0),rtol=0,atol=0)
        np.testing.assert_array_equal(maximum,v.max(axis=0))
    def test_known_gaussian_recovered(self):
        yy,xx=np.mgrid[:50,:50]
        image=gaussian_2d((xx,yy),100,25.3,24.7,1.8,200)
        fits=fit_candidates(image)
        r=fits.loc[fits.retained].iloc[0]
        self.assertAlmostEqual(r.x,25.3,places=5)
        self.assertAlmostEqual(r.y,24.7,places=5)
        self.assertAlmostEqual(r.R,1.8,places=5)
    def test_empty_detection_schema(self):
        for val in [0.,200.]:
            fits=fit_candidates(np.full((32,32),val))
            self.assertTrue(fits.empty);self.assertEqual(list(fits.columns),FIT_COLUMNS)
    def test_stable_ids_survive_rejection(self):
        yy,xx=np.mgrid[:80,:80]
        image=gaussian_2d((xx,yy),300,40,40,3.,200)+gaussian_2d((xx,yy),80,20,20,1.8,0)
        fits=fit_candidates(image,AnalysisConfig(max_R=2.5))
        retained=fits.loc[fits.retained]
        self.assertEqual(list(retained.candidate_id),[2])
        self.assertEqual(fits.iloc[0].reason,'width_outside_range')
    def test_mask_counts_and_subpixel_geometry(self):
        _,a,b=local_masks((40,40),20,20)
        self.assertEqual(a.sum(),29)
        _,a2,b2=local_masks((40,40),20.800828,20.836582)
        self.assertEqual(a2.sum(),28)
        self.assertFalse(np.any(a2 & b2))
    def test_edge_rejection(self):
        with self.assertRaises(ValueError):local_masks((40,40),3,20)
    def test_trace_matches_original_full_frame_formula(self):
        rng=np.random.default_rng(8);v=rng.integers(150,300,(11,40,40),dtype=np.uint16)
        x,y=20.3,20.2
        tr=extract_trace(v,x,y,AnalysisConfig(chunk_frames=3))
        yy,xx=np.mgrid[:40,:40];d2=(xx-x)**2+(yy-y)**2
        a=d2<=9;b=(d2>=25)&(d2<=64)
        signal=v[:,a].mean(axis=1);background=np.median(v[:,b],axis=1)
        np.testing.assert_allclose(tr['signal'],signal,rtol=0,atol=0)
        np.testing.assert_allclose(tr['background'],background,rtol=0,atol=0)
        np.testing.assert_allclose(tr['corrected'],signal-background,rtol=0,atol=0)
        np.testing.assert_allclose(tr['integrated_corrected'],(signal-background)*a.sum())
    def test_saturation_reporting(self):
        v=np.full((3,40,40),200,dtype=np.uint16);v[0,20,20]=4095
        tr=extract_trace(v,20,20,AnalysisConfig(saturation_adu=4095))
        self.assertAlmostEqual(tr['saturated_aperture_fraction'],1/(3*29))
        self.assertTrue(np.isnan(extract_trace(v,20,20)['saturated_aperture_fraction']))
    def test_threshold_uses_corrected_scale(self):
        raw=np.array([200,200,250,250,200,200],float);background=np.full(6,200.)
        runs=threshold_runs(raw-background,25,.05)
        self.assertEqual(list(runs.state),['below_or_equal','above','below_or_equal'])
        self.assertEqual(list(runs.complete),[False,True,False])
        self.assertAlmostEqual(runs.iloc[1].sampled_duration_s,.1)
    def test_dwell_boundary_censoring_and_no_time(self):
        runs=threshold_runs([0,0,10,10,0,0,10],5)
        self.assertEqual(list(runs.n_frames),[2,2,2,1])
        self.assertEqual(list(runs.complete),[False,True,True,False])
        self.assertTrue(runs.sampled_duration_s.isna().all())
        one=threshold_runs([10],5,.05)
        self.assertTrue(one.iloc[0].left_censored and one.iloc[0].right_censored)
        self.assertFalse(one.iloc[0].complete)
    def test_invalid_config(self):
        for cfg in [AnalysisConfig(aperture_radius=6),AnalysisConfig(min_R=8),AnalysisConfig(chunk_frames=0)]:
            with self.assertRaises(ValueError):cfg.validate()

class WorkflowTests(unittest.TestCase):
    def test_synthetic_workflow_and_exports(self):
        v,truth=synthetic_video();cfg=AnalysisConfig()
        mean,_=projections(v);fits=fit_candidates(mean,cfg);ret=fits.loc[fits.retained]
        self.assertEqual(len(ret),2)
        traces={int(r.candidate_id):extract_trace(v,r.x,r.y,cfg) for r in ret.itertuples()}
        for r in ret.itertuples():
            target=min(truth,key=lambda q:np.hypot(q['x']-r.x,q['y']-r.y))
            self.assertLess(np.hypot(target['x']-r.x,target['y']-r.y),.1)
            tr=traces[int(r.candidate_id)]
            np.testing.assert_array_equal(tr['corrected']>25,target['bright_frames'])
        t,dt=timing(len(v),.05,.05)
        with tempfile.TemporaryDirectory() as d:
            p=export_results(Path(d)/'run',fits,traces,t,{'synthetic_demo':True},cfg,{1:25},dt)
            df=pd.read_csv(p/'extracted_traces.csv')
            self.assertEqual(len(df),len(v)*2)
            np.testing.assert_allclose(df.corrected_mean_counts,df.signal_mean_counts-df.background_median_counts,atol=1e-12)
            self.assertEqual(set(df.candidate_id),set(ret.candidate_id))
            with self.assertRaises(FileExistsError):export_results(p,fits,traces,t,{},cfg)
    def test_empty_exports(self):
        fits=fit_candidates(np.zeros((32,32)))
        with tempfile.TemporaryDirectory() as d:
            p=export_results(Path(d)/'empty',fits,{},np.arange(3.),{},AnalysisConfig())
            self.assertTrue(pd.read_csv(p/'extracted_traces.csv').empty)
            self.assertTrue(pd.read_csv(p/'retained_candidates.csv').empty)
            self.assertTrue(pd.read_csv(p/'trace_statistics.csv').empty)
    def test_reject_nonfinite_and_length_errors(self):
        with self.assertRaises(ValueError):threshold_runs([],1)
        with self.assertRaises(ValueError):threshold_runs([0,np.nan],1)
        with self.assertRaises(ValueError):fit_candidates(np.full((20,20),np.nan))

if __name__=='__main__':unittest.main()
