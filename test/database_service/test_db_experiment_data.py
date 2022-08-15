# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

"""Test ExperimentData."""
from test.base import QiskitExperimentsTestCase
import os
from unittest import mock
import copy
from random import randrange
import time
import threading
import json
import re
import uuid

import matplotlib.pyplot as plt
import numpy as np

from qiskit.providers.fake_provider import FakeMelbourneV2
from qiskit.result import Result
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus
from qiskit_ibm_experiment import IBMExperimentService
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.framework import AnalysisResult
from qiskit_experiments.framework import BackendData
from qiskit_experiments.database_service.exceptions import (
    ExperimentDataError,
    ExperimentEntryNotFound,
    ExperimentEntryExists,
)
from qiskit_experiments.database_service.device_component import Qubit
from qiskit_experiments.framework.experiment_data import (
    AnalysisStatus,
    ExperimentStatus,
)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax


class TestDbExperimentData(QiskitExperimentsTestCase):
    """Test the ExperimentData class."""

    def setUp(self):
        super().setUp()
        self.backend = FakeMelbourneV2()

    def test_db_experiment_data_attributes(self):
        """Test DB experiment data attributes."""
        attrs = {
            "job_ids": ["job1"],
            "share_level": "public",
            "figure_names": ["figure1"],
            "notes": "some notes",
        }
        exp_data = ExperimentData(
            backend=self.backend,
            experiment_type="qiskit_test",
            experiment_id="1234",
            tags=["tag1", "tag2"],
            metadata={"foo": "bar"},
            **attrs,
        )
        self.assertEqual(exp_data.backend.name, self.backend.name)
        self.assertEqual(exp_data.experiment_type, "qiskit_test")
        self.assertEqual(exp_data.experiment_id, "1234")
        self.assertEqual(exp_data.tags, ["tag1", "tag2"])
        self.assertEqual(exp_data.metadata["foo"], "bar")
        for key, val in attrs.items():
            self.assertEqual(getattr(exp_data, key), val)

    def test_add_data_dict(self):
        """Test add data in dictionary."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        a_dict = {"counts": {"01": 518}}
        dicts = [{"counts": {"00": 284}}, {"counts": {"00": 14}}]

        exp_data.add_data(a_dict)
        exp_data.add_data(dicts)
        self.assertEqual([a_dict] + dicts, exp_data.data())

    def test_add_data_result(self):
        """Test add result data."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        a_result = self._get_job_result(1)
        results = [self._get_job_result(2), self._get_job_result(3)]

        expected = [a_result.get_counts()]
        for res in results:
            expected.extend(res.get_counts())

        exp_data.add_data(a_result)
        exp_data.add_data(results)
        self.assertEqual(expected, [sdata["counts"] for sdata in exp_data.data()])
        self.assertIn(a_result.job_id, exp_data.job_ids)

    def test_add_data_result_metadata(self):
        """Test add result metadata."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        result1 = self._get_job_result(1, has_metadata=False)
        result2 = self._get_job_result(1, has_metadata=True)

        exp_data.add_data(result1)
        exp_data.add_data(result2)
        self.assertNotIn("metadata", exp_data.data(0))
        self.assertIn("metadata", exp_data.data(1))

    def test_add_data_job(self):
        """Test add job data."""
        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(3)
        jobs = []
        for _ in range(2):
            job = mock.create_autospec(Job, instance=True)
            job.result.return_value = self._get_job_result(2)
            job.status.return_value = JobStatus.DONE
            jobs.append(job)

        expected = a_job.result().get_counts()
        for job in jobs:
            expected.extend(job.result().get_counts())

        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        exp_data.add_jobs(a_job)
        self.assertExperimentDone(exp_data)
        exp_data.add_jobs(jobs)
        self.assertExperimentDone(exp_data)
        self.assertEqual(expected, [sdata["counts"] for sdata in exp_data.data()])
        self.assertIn(a_job.job_id(), exp_data.job_ids)

    def test_add_data_job_callback(self):
        """Test add job data with callback."""

        def _callback(_exp_data):
            self.assertIsInstance(_exp_data, ExperimentData)
            self.assertEqual(
                [dat["counts"] for dat in _exp_data.data()], a_job.result().get_counts()
            )
            exp_data.add_figures(str.encode("hello world"))
            exp_data.add_analysis_results(mock.MagicMock())
            nonlocal called_back
            called_back = True

        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(2)
        a_job.status.return_value = JobStatus.DONE

        called_back = False
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        exp_data.add_jobs(a_job)
        exp_data.add_analysis_callback(_callback)
        self.assertExperimentDone(exp_data)
        self.assertTrue(called_back)

    def test_add_data_callback(self):
        """Test add data with callback."""

        def _callback(_exp_data):
            self.assertIsInstance(_exp_data, ExperimentData)
            nonlocal called_back_count, expected_data, subtests
            expected_data.extend(subtests[called_back_count][1])
            self.assertEqual([dat["counts"] for dat in _exp_data.data()], expected_data)
            called_back_count += 1

        a_result = self._get_job_result(1)
        results = [self._get_job_result(1), self._get_job_result(1)]
        a_dict = {"counts": {"01": 518}}
        dicts = [{"counts": {"00": 284}}, {"counts": {"00": 14}}]

        subtests = [
            (a_result, [a_result.get_counts()]),
            (results, [res.get_counts() for res in results]),
            (a_dict, [a_dict["counts"]]),
            (dicts, [dat["counts"] for dat in dicts]),
        ]

        called_back_count = 0
        expected_data = []
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")

        for data, _ in subtests:
            with self.subTest(data=data):
                exp_data.add_data(data)
                exp_data.add_analysis_callback(_callback)
                self.assertExperimentDone(exp_data)

        self.assertEqual(len(subtests), called_back_count)

    def test_add_data_job_callback_kwargs(self):
        """Test add job data with callback and additional arguments."""

        def _callback(_exp_data, **kwargs):
            self.assertIsInstance(_exp_data, ExperimentData)
            self.assertEqual({"foo": callback_kwargs}, kwargs)
            nonlocal called_back
            called_back = True

        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(2)
        a_job.status.return_value = JobStatus.DONE

        called_back = False
        callback_kwargs = "foo"
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        exp_data.add_jobs(a_job)
        exp_data.add_analysis_callback(_callback, foo=callback_kwargs)
        self.assertExperimentDone(exp_data)
        self.assertTrue(called_back)

    def test_add_data_pending_post_processing(self):
        """Test add job data while post processing is still running."""

        def _callback(_exp_data, **kwargs):
            kwargs["event"].wait(timeout=3)

        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(2)
        a_job.status.return_value = JobStatus.DONE

        event = threading.Event()
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_analysis_callback(_callback, event=event)
        exp_data.add_jobs(a_job)
        with self.assertLogs("qiskit_experiments", "WARNING"):
            exp_data.add_data({"foo": "bar"})

    def test_get_data(self):
        """Test getting data."""
        data1 = []
        for _ in range(5):
            data1.append({"counts": {"00": randrange(1024)}})
        results = self._get_job_result(3)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_data(data1)
        exp_data.add_data(results)
        self.assertEqual(data1[1], exp_data.data(1))
        self.assertEqual(data1[2:4], exp_data.data(slice(2, 4)))
        self.assertEqual(
            results.get_counts(), [sdata["counts"] for sdata in exp_data.data(results.job_id)]
        )

    def test_add_figure(self):
        """Test adding a new figure."""
        hello_bytes = str.encode("hello world")
        file_name = uuid.uuid4().hex
        self.addCleanup(os.remove, file_name)
        with open(file_name, "wb") as file:
            file.write(hello_bytes)

        sub_tests = [
            ("file name", file_name, None),
            ("file bytes", hello_bytes, None),
            ("new name", hello_bytes, "hello_again.svg"),
        ]

        for name, figure, figure_name in sub_tests:
            with self.subTest(name=name):
                exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
                fn = exp_data.add_figures(figure, figure_name)
                self.assertEqual(hello_bytes, exp_data.figure(fn).figure)

    def test_add_figure_plot(self):
        """Test adding a matplotlib figure."""
        figure, ax = plt.subplots()
        ax.plot([1, 2, 3])

        service = self._set_mock_service()
        exp_data = ExperimentData(
            backend=self.backend, experiment_type="qiskit_test", service=service
        )
        exp_data.add_figures(figure, save_figure=True)
        self.assertEqual(figure, exp_data.figure(0).figure)
        service.create_or_update_figure.assert_called_once()
        _, kwargs = service.create_or_update_figure.call_args
        self.assertIsInstance(kwargs["figure"], bytes)

    def test_add_figures(self):
        """Test adding multiple new figures."""
        hello_bytes = [str.encode("hello world"), str.encode("hello friend")]
        file_names = [uuid.uuid4().hex, uuid.uuid4().hex]
        for idx, fn in enumerate(file_names):
            self.addCleanup(os.remove, fn)
            with open(fn, "wb") as file:
                file.write(hello_bytes[idx])

        sub_tests = [
            ("file names", file_names, None),
            ("file bytes", hello_bytes, None),
            ("new names", hello_bytes, ["hello1.svg", "hello2.svg"]),
        ]

        for name, figures, figure_names in sub_tests:
            with self.subTest(name=name):
                exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
                added_names = exp_data.add_figures(figures, figure_names)
                for idx, added_fn in enumerate(added_names):
                    self.assertEqual(hello_bytes[idx], exp_data.figure(added_fn).figure)

    def test_add_figure_overwrite(self):
        """Test updating an existing figure."""
        hello_bytes = str.encode("hello world")
        friend_bytes = str.encode("hello friend!")

        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        fn = exp_data.add_figures(hello_bytes)
        with self.assertRaises(ExperimentEntryExists):
            exp_data.add_figures(friend_bytes, fn)

        exp_data.add_figures(friend_bytes, fn, overwrite=True)
        self.assertEqual(friend_bytes, exp_data.figure(fn).figure)

    def test_add_figure_save(self):
        """Test saving a figure in the database."""
        hello_bytes = str.encode("hello world")
        service = self._set_mock_service()
        exp_data = ExperimentData(
            backend=self.backend, experiment_type="qiskit_test", service=service
        )
        exp_data.add_figures(hello_bytes, save_figure=True)
        service.create_or_update_figure.assert_called_once()
        _, kwargs = service.create_or_update_figure.call_args
        self.assertEqual(kwargs["figure"], hello_bytes)
        self.assertEqual(kwargs["experiment_id"], exp_data.experiment_id)

    def test_add_figure_metadata(self):
        hello_bytes = str.encode("hello world")
        qubits = [0, 1, 2]
        exp_data = ExperimentData(
            backend=self.backend,
            experiment_type="qiskit_test",
            metadata={"physical_qubits": qubits},
        )
        exp_data.add_figures(hello_bytes)
        exp_data.figure(0).metadata["foo"] = "bar"
        figure_data = exp_data.figure(0)

        self.assertEqual(figure_data.metadata["qubits"], qubits)
        self.assertEqual(figure_data.metadata["foo"], "bar")
        expected_name_prefix = "qiskit_test_Fig-0_Exp-"
        self.assertEqual(figure_data.name[: len(expected_name_prefix)], expected_name_prefix)

        exp_data2 = ExperimentData(
            backend=self.backend,
            experiment_type="qiskit_test",
            metadata={"physical_qubits": [1, 2, 3, 4]},
        )
        exp_data2.add_figures(figure_data, "new_name.svg")
        figure_data = exp_data2.figure("new_name.svg")

        # metadata should not change when adding to new ExperimentData
        self.assertEqual(figure_data.metadata["qubits"], qubits)
        self.assertEqual(figure_data.metadata["foo"], "bar")
        # name should change
        self.assertEqual(figure_data.name, "new_name.svg")

        # can set the metadata to new dictionary
        figure_data.metadata = {"bar": "foo"}
        self.assertEqual(figure_data.metadata["bar"], "foo")

        # cannot set the metadata to something other than dictionary
        with self.assertRaises(ValueError):
            figure_data.metadata = ["foo", "bar"]

    def test_add_figure_bad_input(self):
        """Test adding figures with bad input."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        self.assertRaises(ValueError, exp_data.add_figures, ["foo", "bar"], ["name"])

    def test_get_figure(self):
        """Test getting figure."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        figure_template = "hello world {}"
        name_template = "figure_{}.svg"
        for idx in range(3):
            exp_data.add_figures(
                str.encode(figure_template.format(idx)), figure_names=name_template.format(idx)
            )
        idx = randrange(3)
        expected_figure = str.encode(figure_template.format(idx))
        self.assertEqual(expected_figure, exp_data.figure(name_template.format(idx)).figure)
        self.assertEqual(expected_figure, exp_data.figure(idx).figure)

        file_name = uuid.uuid4().hex
        self.addCleanup(os.remove, file_name)
        exp_data.figure(idx, file_name)
        with open(file_name, "rb") as file:
            self.assertEqual(expected_figure, file.read())

    def test_delete_figure(self):
        """Test deleting a figure."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        id_template = "figure_{}.svg"
        for idx in range(3):
            exp_data.add_figures(str.encode("hello world"), id_template.format(idx))

        sub_tests = [(1, id_template.format(1)), (id_template.format(2), id_template.format(2))]

        for del_key, figure_name in sub_tests:
            with self.subTest(del_key=del_key):
                exp_data.delete_figure(del_key)
                self.assertRaises(ExperimentEntryNotFound, exp_data.figure, figure_name)

    def test_delayed_backend(self):
        """Test initializing experiment data without a backend."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        self.assertIsNone(exp_data.backend)
        exp_data.save_metadata()
        a_job = mock.create_autospec(Job, instance=True)
        exp_data.add_jobs(a_job)
        self.assertIsNotNone(exp_data.backend)

    def test_different_backend(self):
        """Test setting a different backend."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        a_job = mock.create_autospec(Job, instance=True)
        self.assertNotEqual(exp_data.backend, a_job.backend())
        with self.assertLogs("qiskit_experiments", "WARNING"):
            exp_data.add_jobs(a_job)

    def test_add_get_analysis_result(self):
        """Test adding and getting analysis results."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        results = []
        for idx in range(5):
            res = mock.MagicMock()
            res.result_id = idx
            results.append(res)
            exp_data.add_analysis_results(res)

        self.assertEqual(results, exp_data.analysis_results())
        self.assertEqual(results[1], exp_data.analysis_results(1))
        self.assertEqual(results[2:4], exp_data.analysis_results(slice(2, 4)))
        self.assertEqual(results[4], exp_data.analysis_results(results[4].result_id))

    def test_add_get_analysis_results(self):
        """Test adding and getting a list of analysis results."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        results = []
        for idx in range(5):
            res = mock.MagicMock()
            res.result_id = idx
            results.append(res)
        exp_data.add_analysis_results(results)

        self.assertEqual(results, exp_data.analysis_results())

    def test_delete_analysis_result(self):
        """Test deleting analysis result."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        id_template = "result_{}"
        for idx in range(3):
            res = mock.MagicMock()
            res.result_id = id_template.format(idx)
            exp_data.add_analysis_results(res)

        subtests = [(0, id_template.format(0)), (id_template.format(2), id_template.format(2))]
        for del_key, res_id in subtests:
            with self.subTest(del_key=del_key):
                exp_data.delete_analysis_result(del_key)
                self.assertRaises(ExperimentEntryNotFound, exp_data.analysis_results, res_id)

    def test_save_metadata(self):
        """Test saving experiment metadata."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        service = mock.create_autospec(IBMExperimentService, instance=True)
        exp_data.service = service
        exp_data.save_metadata()
        service.create_or_update_experiment.assert_called_once()
        data = service.create_or_update_experiment.call_args[0][0]
        self.assertEqual(exp_data.experiment_id, data.experiment_id)

    def test_save(self):
        """Test saving all experiment related data."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        service = mock.create_autospec(IBMExperimentService, instance=True)
        exp_data.add_figures(str.encode("hello world"))
        analysis_result = mock.MagicMock()
        exp_data.add_analysis_results(analysis_result)
        exp_data.service = service
        exp_data.save()
        service.create_or_update_experiment.assert_called_once()
        service.create_or_update_figure.assert_called_once()
        analysis_result.save.assert_called_once()

    def test_save_delete(self):
        """Test saving all deletion."""
        exp_data = ExperimentData(backend=self.backend, experiment_type="qiskit_test")
        service = mock.create_autospec(IBMExperimentService, instance=True)
        exp_data.add_figures(str.encode("hello world"))
        exp_data.add_analysis_results(mock.MagicMock())
        exp_data.delete_analysis_result(0)
        exp_data.delete_figure(0)
        exp_data.service = service

        exp_data.save()
        service.create_or_update_experiment.assert_called_once()
        service.delete_figure.assert_called_once()
        service.delete_analysis_result.assert_called_once()

    def test_set_service_direct(self):
        """Test setting service directly."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        self.assertIsNone(exp_data.service)
        mock_service = mock.MagicMock()
        exp_data.service = mock_service
        self.assertEqual(mock_service, exp_data.service)

        with self.assertRaises(ExperimentDataError):
            exp_data.service = mock_service

    def test_auto_save(self):
        """Test auto save."""
        service = self._set_mock_service()
        exp_data = ExperimentData(
            backend=self.backend, experiment_type="qiskit_test", service=service
        )
        exp_data.auto_save = True
        mock_result = mock.MagicMock()

        subtests = [
            # update function, update parameters, service called
            (exp_data.add_analysis_results, (mock_result,), mock_result.save),
            (exp_data.add_figures, (str.encode("hello world"),), service.create_or_update_figure),
            (exp_data.delete_figure, (0,), service.delete_figure),
            (exp_data.delete_analysis_result, (0,), service.delete_analysis_result),
            (setattr, (exp_data, "tags", ["foo"]), service.create_or_update_experiment),
            (setattr, (exp_data, "notes", "foo"), service.create_or_update_experiment),
            (setattr, (exp_data, "share_level", "hub"), service.create_or_update_experiment),
        ]

        for func, params, called in subtests:
            with self.subTest(func=func):
                func(*params)
                if called:
                    called.assert_called_once()
                service.reset_mock()

    def test_status_job_pending(self):
        """Test experiment status when job is pending."""
        job1 = mock.create_autospec(Job, instance=True)
        job1.result.return_value = self._get_job_result(3)
        job1.status.return_value = JobStatus.DONE

        event = threading.Event()
        job2 = mock.create_autospec(Job, instance=True)
        job2.result = lambda *args, **kwargs: event.wait(timeout=15)
        job2.status = lambda: JobStatus.CANCELLED if event.is_set() else JobStatus.RUNNING
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job1)
        exp_data.add_jobs(job2)
        exp_data.add_analysis_callback(lambda *args, **kwargs: event.wait(timeout=15))
        self.assertEqual(ExperimentStatus.RUNNING, exp_data.status())
        self.assertEqual(JobStatus.RUNNING, exp_data.job_status())
        self.assertEqual(AnalysisStatus.QUEUED, exp_data.analysis_status())

        # Cleanup
        with self.assertLogs("qiskit_experiments", "WARNING"):
            event.set()
            exp_data.block_for_results()

    def test_status_job_error(self):
        """Test experiment status when job failed."""
        job1 = mock.create_autospec(Job, instance=True)
        job1.result.return_value = self._get_job_result(3)
        job1.status.return_value = JobStatus.DONE

        job2 = mock.create_autospec(Job, instance=True)
        job2.status.return_value = JobStatus.ERROR

        exp_data = ExperimentData(experiment_type="qiskit_test")
        with self.assertLogs(logger="qiskit_experiments.framework", level="WARN") as cm:
            exp_data.add_jobs([job1, job2])
        self.assertIn("Adding a job from a backend", ",".join(cm.output))
        self.assertEqual(ExperimentStatus.ERROR, exp_data.status())

    def test_status_post_processing(self):
        """Test experiment status during post processing."""
        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)
        job.status.return_value = JobStatus.DONE

        event = threading.Event()
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback((lambda *args, **kwargs: event.wait(timeout=15)))
        status = exp_data.status()
        self.assertEqual(ExperimentStatus.POST_PROCESSING, status)

    def test_status_cancelled_analysis(self):
        """Test experiment status during post processing."""
        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)
        job.status.return_value = JobStatus.DONE

        event = threading.Event()
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback((lambda *args, **kwargs: event.wait(timeout=2)))
        # Add second callback because the first can't be cancelled once it has started
        exp_data.add_analysis_callback((lambda *args, **kwargs: event.wait(timeout=20)))
        exp_data.cancel_analysis()
        status = exp_data.status()
        self.assertEqual(ExperimentStatus.CANCELLED, status)

    def test_status_post_processing_error(self):
        """Test experiment status when post processing failed."""

        def _post_processing(*args, **kwargs):
            raise ValueError("Kaboom!")

        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)
        job.status.return_value = JobStatus.DONE

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        with self.assertLogs(logger="qiskit_experiments.framework", level="WARN") as cm:
            exp_data.add_jobs(job)
            exp_data.add_analysis_callback(_post_processing)
            exp_data.block_for_results()
        self.assertEqual(ExperimentStatus.ERROR, exp_data.status())
        self.assertIn("Kaboom!", ",".join(cm.output))

    def test_status_done(self):
        """Test experiment status when all jobs are done."""
        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)
        job.status.return_value = JobStatus.DONE
        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_jobs(job)
        self.assertExperimentDone(exp_data)
        self.assertEqual(ExperimentStatus.DONE, exp_data.status())

    def test_set_tags(self):
        """Test updating experiment tags."""
        exp_data = ExperimentData(experiment_type="qiskit_test", tags=["foo"])
        self.assertEqual(["foo"], exp_data.tags)
        exp_data.tags = ["bar"]
        self.assertEqual(["bar"], exp_data.tags)

    def test_cancel_jobs(self):
        """Test canceling experiment jobs."""
        event = threading.Event()
        cancel_count = 0

        def _job_result():
            event.wait(timeout=15)
            raise ValueError("Job was cancelled.")

        def _job_cancel():
            nonlocal cancel_count
            cancel_count += 1
            event.set()

        exp_data = ExperimentData(experiment_type="qiskit_test")
        event = threading.Event()
        self.addCleanup(event.set)
        job = mock.create_autospec(Job, instance=True)
        job.job_id.return_value = "1234"
        job.cancel = _job_cancel
        job.result = _job_result
        job.status = lambda: JobStatus.CANCELLED if event.is_set() else JobStatus.RUNNING
        exp_data.add_jobs(job)

        with self.assertLogs("qiskit_experiments", "WARNING"):
            exp_data.cancel_jobs()
            self.assertEqual(cancel_count, 1)
            self.assertEqual(exp_data.job_status(), JobStatus.CANCELLED)
            self.assertEqual(exp_data.status(), ExperimentStatus.CANCELLED)

    def test_cancel_analysis(self):
        """Test canceling experiment analysis."""

        event = threading.Event()
        self.addCleanup(event.set)

        def _job_result():
            event.wait(timeout=15)
            return self._get_job_result(1)

        def _analysis(*args):  # pylint: disable = unused-argument
            event.wait(timeout=15)

        job = mock.create_autospec(Job, instance=True)
        job.job_id.return_value = "1234"
        job.result = _job_result
        job.status = lambda: JobStatus.DONE if event.is_set() else JobStatus.RUNNING

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback(_analysis)
        exp_data.cancel_analysis()

        # Test status while job still running
        self.assertEqual(exp_data.job_status(), JobStatus.RUNNING)
        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.CANCELLED)
        self.assertEqual(exp_data.status(), ExperimentStatus.RUNNING)

        # Test status after job finishes
        event.set()
        self.assertEqual(exp_data.job_status(), JobStatus.DONE)
        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.CANCELLED)
        self.assertEqual(exp_data.status(), ExperimentStatus.CANCELLED)

    def test_partial_cancel_analysis(self):
        """Test canceling experiment analysis."""

        event = threading.Event()
        self.addCleanup(event.set)
        run_analysis = []

        def _job_result():
            event.wait(timeout=3)
            return self._get_job_result(1)

        def _analysis(expdata, name=None, timeout=0):  # pylint: disable = unused-argument
            event.wait(timeout=timeout)
            run_analysis.append(name)

        job = mock.create_autospec(Job, instance=True)
        job.job_id.return_value = "1234"
        job.result = _job_result
        job.status = lambda: JobStatus.DONE if event.is_set() else JobStatus.RUNNING

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback(_analysis, name=1, timeout=1)
        exp_data.add_analysis_callback(_analysis, name=2, timeout=30)
        cancel_id = exp_data._analysis_callbacks.keys()[-1]
        exp_data.add_analysis_callback(_analysis, name=3, timeout=1)
        consequent_cancel_id = exp_data._analysis_callbacks.keys()[-1]
        exp_data.cancel_analysis(cancel_id)

        # Test status while job is still running
        self.assertEqual(exp_data.job_status(), JobStatus.RUNNING)
        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.CANCELLED)
        self.assertEqual(exp_data.status(), ExperimentStatus.RUNNING)

        # Test status after job finishes
        event.set()
        self.assertEqual(exp_data.job_status(), JobStatus.DONE)
        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.CANCELLED)
        self.assertEqual(exp_data.status(), ExperimentStatus.CANCELLED)

        # Check that correct analysis callback was cancelled
        exp_data.block_for_results()
        self.assertEqual(run_analysis, [1])
        for cid, analysis in exp_data._analysis_callbacks.items():
            if cid in [cancel_id, consequent_cancel_id]:
                self.assertEqual(analysis.status, AnalysisStatus.CANCELLED)
            else:
                self.assertEqual(analysis.status, AnalysisStatus.DONE)

    def test_cancel(self):
        """Test canceling experiment jobs and analysis."""

        event = threading.Event()
        self.addCleanup(event.set)

        def _job_result():
            event.wait(timeout=15)
            raise ValueError("Job was cancelled.")

        def _analysis(*args):  # pylint: disable = unused-argument
            event.wait(timeout=15)

        def _status():
            if event.is_set():
                return JobStatus.CANCELLED
            return JobStatus.RUNNING

        job = mock.create_autospec(Job, instance=True)
        job.job_id.return_value = "1234"
        job.result = _job_result
        job.cancel = event.set
        job.status = _status

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback(_analysis)
        exp_data.cancel()

        # Test status while job still running
        self.assertEqual(exp_data.job_status(), JobStatus.CANCELLED)
        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.CANCELLED)
        self.assertEqual(exp_data.status(), ExperimentStatus.CANCELLED)

    def test_add_jobs_timeout(self):
        """Test timeout kwarg of add_jobs"""

        event = threading.Event()
        self.addCleanup(event.set)

        def _job_result():
            event.wait(timeout=15)
            raise ValueError("Job was cancelled.")

        job = mock.create_autospec(Job, instance=True)
        job.job_id.return_value = "1234"
        job.result = _job_result
        job.cancel = event.set
        job.status = lambda: JobStatus.CANCELLED if event.is_set() else JobStatus.RUNNING

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job, timeout=0.5)

        with self.assertLogs("qiskit_experiments", "WARNING"):
            exp_data.block_for_results()
            self.assertEqual(exp_data.job_status(), JobStatus.CANCELLED)
            self.assertEqual(exp_data.status(), ExperimentStatus.CANCELLED)

    def test_metadata_serialization(self):
        """Test experiment metadata serialization."""
        metadata = {"complex": 2 + 3j, "numpy": np.zeros(2)}
        exp_data = ExperimentData(experiment_type="qiskit_test", metadata=metadata)
        serialized = json.dumps(exp_data.metadata, cls=exp_data._json_encoder)
        self.assertIsInstance(serialized, str)
        self.assertTrue(json.loads(serialized))

        deserialized = json.loads(serialized, cls=exp_data._json_decoder)
        self.assertEqual(metadata["complex"], deserialized["complex"])
        self.assertEqual(metadata["numpy"].all(), deserialized["numpy"].all())

    def test_errors(self):
        """Test getting experiment error message."""

        def _post_processing(*args, **kwargs):  # pylint: disable=unused-argument
            raise ValueError("Kaboom!")

        job1 = mock.create_autospec(Job, instance=True)
        job1.job_id.return_value = "1234"
        job1.status.return_value = JobStatus.DONE

        job2 = mock.create_autospec(Job, instance=True)
        job2.status.return_value = JobStatus.ERROR
        job2.job_id.return_value = "5678"

        exp_data = ExperimentData(experiment_type="qiskit_test")
        with self.assertLogs(logger="qiskit_experiments.framework", level="WARN") as cm:
            exp_data.add_jobs(job1)
            exp_data.add_analysis_callback(_post_processing)
            exp_data.add_jobs(job2)
            exp_data.block_for_results()
        self.assertEqual(ExperimentStatus.ERROR, exp_data.status())
        self.assertIn("Kaboom", ",".join(cm.output))
        self.assertTrue(re.match(r".*5678.*Kaboom!", exp_data.errors(), re.DOTALL))

    def test_simple_methods_from_callback(self):
        """Test that simple methods used in call back function don't hang

        This test runs through many of the public methods of ExperimentData
        from analysis callbacks to make sure that they do not raise exceptions
        or hang the analysis thread. Hangs have occurred in the past when one
        of these methods blocks waiting for analysis to complete.

        These methods are not tested because they explicitly assume they are
        run from the main thread:

            + copy
            + block_for_results

        These methods are not tested because they require additional setup.
        They could be tested in separate tests:

            + save
            + save_metadata
            + add_jobs
            + cancel
            + cancel_analysis
            + cancel_jobs
        """

        def callback1(exp_data):
            """Callback function that call add_analysis_callback"""
            exp_data.add_analysis_callback(callback2)
            result = AnalysisResult("result_name", 0, [Qubit(0)], "experiment_id")
            exp_data.add_analysis_results(result)
            figure = get_non_gui_ax().get_figure()
            exp_data.add_figures(figure, "figure.svg")
            exp_data.add_data({"key": 1.2})
            exp_data.data()

        def callback2(exp_data):
            """Callback function that exercises status lookups"""
            exp_data.figure("figure.svg")
            exp_data.jobs()

            exp_data.analysis_results("result_name", block=False)

            exp_data.delete_figure("figure.svg")
            exp_data.delete_analysis_result("result_name")

            exp_data.status()
            exp_data.job_status()
            exp_data.analysis_status()

            exp_data.errors()
            exp_data.job_errors()
            exp_data.analysis_errors()

        exp_data = ExperimentData(experiment_type="qiskit_test")

        exp_data.add_analysis_callback(callback1)
        exp_data.block_for_results(timeout=3)

        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.DONE)

    def test_recursive_callback_raises(self):
        """Test handling of excepting callbacks"""

        def callback1(exp_data):
            """Callback function that calls add_analysis_callback"""
            time.sleep(1)
            exp_data.add_analysis_callback(callback2)
            result = AnalysisResult("RESULT1", True, ["Q0"], exp_data.experiment_id)
            exp_data.add_analysis_results(result)

        def callback2(exp_data):
            """Callback function that exercises status lookups"""
            time.sleep(1)
            exp_data.add_analysis_callback(callback3)
            raise RuntimeError("YOU FAIL")

        def callback3(exp_data):
            """Callback function that exercises status lookups"""
            time.sleep(1)
            result = AnalysisResult("RESULT2", True, ["Q0"], exp_data.experiment_id)
            exp_data.add_analysis_results(result)

        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_analysis_callback(callback1)
        exp_data.block_for_results(timeout=10)
        results = exp_data.analysis_results(block=False)

        self.assertEqual(exp_data.analysis_status(), AnalysisStatus.ERROR)
        self.assertTrue("RuntimeError: YOU FAIL" in exp_data.analysis_errors())
        self.assertEqual(len(results), 1)

    def test_source(self):
        """Test getting experiment source."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        self.assertIn("ExperimentData", exp_data.source["class"])
        self.assertTrue(exp_data.source["qiskit_version"])

    def test_block_for_jobs(self):
        """Test blocking for jobs."""

        def _sleeper(*args, **kwargs):  # pylint: disable=unused-argument
            time.sleep(2)
            nonlocal sleep_count
            sleep_count += 1
            return self._get_job_result(1)

        sleep_count = 0
        job = mock.create_autospec(Job, instance=True)
        job.result = _sleeper
        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_jobs(job)
        exp_data.add_analysis_callback(_sleeper)
        self.assertExperimentDone(exp_data)
        self.assertEqual(2, sleep_count)

    def test_additional_attr(self):
        """Test additional experiment attributes."""
        exp_data = ExperimentData(experiment_type="qiskit_test", foo="foo")
        self.assertEqual("foo", exp_data.foo)

    def test_copy_metadata(self):
        """Test copy metadata."""
        exp_data = ExperimentData(experiment_type="qiskit_test")
        exp_data.add_data(self._get_job_result(1))
        result = mock.MagicMock()
        exp_data.add_analysis_results(result)
        copied = exp_data.copy(copy_results=False)
        self.assertEqual(exp_data.data(), copied.data())
        self.assertFalse(copied.analysis_results())

    def test_copy_metadata_pending_job(self):
        """Test copy metadata with a pending job."""
        event = threading.Event()
        self.addCleanup(event.set)
        job_results1 = self._get_job_result(1)
        job_results2 = self._get_job_result(1)

        def _job1_result():
            event.wait(timeout=15)
            return job_results1

        def _job2_result():
            event.wait(timeout=15)
            return job_results2

        exp_data = ExperimentData(experiment_type="qiskit_test")
        job = mock.create_autospec(Job, instance=True)
        job.result = _job1_result
        exp_data.add_jobs(job)

        copied = exp_data.copy(copy_results=False)
        job2 = mock.create_autospec(Job, instance=True)
        job2.result = _job2_result
        copied.add_jobs(job2)
        event.set()

        exp_data.block_for_results()
        copied.block_for_results()

        self.assertEqual(1, len(exp_data.data()))
        self.assertEqual(2, len(copied.data()))
        self.assertIn(
            exp_data.data(0)["counts"], [copied.data(0)["counts"], copied.data(1)["counts"]]
        )

    def _get_job_result(self, circ_count, has_metadata=False):
        """Return a job result with random counts."""
        job_result = {
            "backend_name": BackendData(self.backend).name,
            "backend_version": "1.1.1",
            "qobj_id": "1234",
            "job_id": "some_job_id",
            "success": True,
            "results": [],
        }
        circ_result_template = {"shots": 1024, "success": True, "data": {}}

        for _ in range(circ_count):
            counts = randrange(1024)
            circ_result = copy.copy(circ_result_template)
            circ_result["data"] = {"counts": {"0x0": counts, "0x3": 1024 - counts}}
            if has_metadata:
                circ_result["header"] = {"metadata": {"meas_basis": "pauli"}}
            job_result["results"].append(circ_result)

        return Result.from_dict(job_result)

    def _set_mock_service(self):
        """Add a mock service to the backend."""
        mock_provider = mock.MagicMock()
        self.backend._provider = mock_provider
        mock_service = mock.create_autospec(IBMExperimentService, instance=True)
        mock_provider.service.return_value = mock_service
        return mock_service
