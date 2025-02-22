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
"""
Quantum State Tomography experiment
"""

from typing import Union, Optional, Iterable, List, Sequence
from qiskit.providers.backend import Backend
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

from qiskit_experiments.warnings import deprecate_arguments
from qiskit_experiments.exceptions import QiskitError
from .tomography_experiment import TomographyExperiment, TomographyAnalysis, BaseAnalysis
from .qst_analysis import StateTomographyAnalysis
from . import basis


class StateTomography(TomographyExperiment):
    """Quantum state tomography experiment.

    # section: overview
        Quantum state tomography (QST) is a method for experimentally
        reconstructing the quantum state from measurement data.

        A QST experiment measures the state prepared by quantum
        circuit in different measurement bases and post-processes the
        measurement data to reconstruct the state.

    # section: note
        Performing full state tomography on an `N`-qubit state requires
        running :math:`3^N` measurement circuits when using the default
        measurement basis.

    # section: analysis_ref
        :py:class:`StateTomographyAnalysis`

    # section: see_also
        qiskit_experiments.library.tomography.tomography_experiment.TomographyExperiment

    """

    @deprecate_arguments({"qubits": "physical_qubits"}, "0.5")
    def __init__(
        self,
        circuit: Union[QuantumCircuit, Instruction, BaseOperator, Statevector],
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_indices: Optional[Sequence[int]] = None,
        measurement_qubits: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[List[int]]] = None,
        analysis: Union[BaseAnalysis, None, str] = "default",
        target: Union[Statevector, DensityMatrix, None, str] = "default",
    ):
        """Initialize a quantum process tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            backend: The backend to run the experiment on.
            physical_qubits: Optional, the physical qubits for the initial state circuit.
                If None this will be qubits [0, N) for an N-qubit circuit.
            measurement_basis: Tomography basis for measurements. If not specified the
                default basis is the :class:`~basis.PauliMeasurementBasis`.
            measurement_indices: Optional, the `physical_qubits` indices to be measured.
                If None all circuit physical qubits will be measured.
            measurement_qubits: DEPRECATED, equivalent to measurement_indices.
            basis_indices: Optional, a list of basis indices for generating partial
                tomography measurement data. Each item should be given as a list of
                measurement basis configurations ``[m[0], m[1], ...]`` where ``m[i]``
                is the measurement basis index for qubit-i. If not specified full
                tomography for all indices of the measurement basis will be performed.
            analysis: Optional, a custom analysis instance to use. If ``"default"``
                :class:`~.StateTomographyAnalysis` will be used. If None no analysis
                instance will be set.
            target: Optional, a custom quantum state target for computing the
                state fidelity of the fitted density matrix during analysis.
                If "default" the state will be inferred from the input circuit
                if it contains no classical instructions.
        """
        if isinstance(circuit, Statevector):
            # Convert to circuit using initialize instruction
            circ = QuantumCircuit(circuit.num_qubits)
            circ.initialize(circuit)
            circuit = circ

        if basis_indices is not None:
            # Add trivial preparation indices for base class
            basis_indices = [([], i) for i in basis_indices]

        if analysis == "default":
            analysis = StateTomographyAnalysis()

        super().__init__(
            circuit,
            backend=backend,
            physical_qubits=physical_qubits,
            measurement_basis=measurement_basis,
            measurement_indices=measurement_indices,
            measurement_qubits=measurement_qubits,
            basis_indices=basis_indices,
            analysis=analysis,
        )

        # Set target quantum state
        if isinstance(self.analysis, TomographyAnalysis):
            if target == "default":
                target = self._target_quantum_state()
            self.analysis.set_options(target=target)

    def _target_quantum_state(self) -> Union[Statevector, DensityMatrix]:
        """Return the state tomography target"""
        # Check if circuit contains measure instructions
        # If so we cannot return target state
        circuit_ops = self._circuit.count_ops()
        if "measure" in circuit_ops:
            return None

        try:
            circuit = self._permute_circuit()
            if "reset" in circuit_ops or "kraus" in circuit_ops or "superop" in circuit_ops:
                state = DensityMatrix(circuit)
            else:
                state = Statevector(circuit)
        except QiskitError:
            # Circuit couldn't be simulated
            return None

        if not self._meas_indices:
            return state

        non_meas_qargs = list(range(len(self._meas_indices), self._circuit.num_qubits))
        if non_meas_qargs:
            # Trace over non-measured qubits
            state = partial_trace(state, non_meas_qargs)

        return state
