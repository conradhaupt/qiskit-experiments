from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from uncertainties import unumpy as unp


class BasePlotter(ABC):
    def __init__(
        self,
        figure_name: str,
    ):
        """An abstract class for data plotters.

        Args:
            figure_name: The name of the figure to be plotted.
        """
        self._figure_name = figure_name

    @property
    def figure_name(self) -> str:
        return self._figure_name

    @figure_name.setter
    def figure_name(self, new_figure_name: str):
        self._figure_name = new_figure_name

    def __call__(self, data: np.ndarray) -> Tuple[Figure, str]:
        """Plots ``data``.

        Calls :py:meth:`_plot_figure`, which should be overridden by subclasses of
        :py:class:`BasePlotter`. Subclasses must not override :py:meth:`__call__`.

        Args:
            data: The data to be plotted.

        Returns:
            tuple: The plotted figure and the figure name.
        """
        fig = self._plot_figure(data)
        return fig, self.figure_name

    @classmethod
    def _identify_data(cls, data: np.ndarray) -> Tuple[MeasLevel, MeasReturnType]:
        """Identifies the type of data stored in ``data``.

        Args:
            data: The data being identified/

        Raises:
            QiskitError: If the data isn't three or four dimensional.
            QiskitError: If the data doesn't contain IQ values (i.e., data.shape[-1]=2).

        Returns:
            tuple: A tuple containing the measurement level and return type associated with ``data``.
        """
        # Check if the data is kerneled single-shot or averaged
        if len(data.shape) not in [3, 4]:
            raise QiskitError(f"Incorrect data dimensions: expected 4 or 3, got {len(data.shape)}")

        # Check if the data contains IQ values (i.e., I and Q values in last dimension)
        if data.shape[-1] != 2:
            raise QiskitError(f"Data does not contain IQ data")

        # Determine measurement level and type
        if len(data.shape) == 4:
            return MeasLevel.KERNELED, MeasReturnType.SINGLE
        else:
            return MeasLevel.KERNELED, MeasReturnType.AVERAGE

    @abstractmethod
    def _plot_figure(self, data: np.ndarray) -> Figure:
        pass


class IQPlotter(BasePlotter):
    def __init__(
        self,
        split_by: Optional[str] = "circuit",
        colour_by: Optional[str] = "memslot",
        n_cols: int = 2,
        axis_width: float = 2,
        axis_height: float = 2,
    ):
        """Plotter for IQ data (single-shot and averaged).

        :py:attr:`split_by` dictates how to split the data into separate axes. :py:attr:`colour_by` does
        the same but for individual series in each axis. Valid values are ``None``, ``"memslot"``, and
        ``""circuit"``. If ``None``, then all IQ points are plotted on the same axis. If ``"memslot"`` or
        ``"circuit"`` is provided, then there will be a figure axis per memory slot or circuit,
        respectively.

        Args:
            split_by: How to split the data into separate axes. Defaults to "circuit".
            colour_by: How to split the data into different series (colours). Defaults to "memslot".
            n_cols: Number of columns of axes to plot, when split_by is not ``None``.
            axis_width: Width of a single axis, in inches. Used to determine the figure size. Defaults to
                2.
            axis_height: Height of a single axis, in inches. USed to determine the figure height.
                Defaults to 2.
        """
        super().__init__("IQ")
        if split_by is not None and split_by == colour_by:
            raise RuntimeError("`split_by` and `colour_by` cannot be equal if they are not None")
        self._split_by = split_by
        self._colour_by = colour_by
        self._n_cols = n_cols
        self._axis_width = axis_width
        self._axis_height = axis_height

    @property
    def split_by(self) -> Optional[str]:
        return self._split_by

    @property
    def colour_by(self) -> Optional[str]:
        return self._colour_by

    def _plot_figure(self, data: np.ndarray) -> Figure:
        """Plots ``data`` as a scatter plot of IQ points.

        Args:
            data: IQ data to be plotted, either single-shot or averaged.

        Returns:
            Figure: The plotted IQ figure.
        """
        meas_level, meas_type = self._identify_data(data)
        if self.split_by == "memslot":
            n_axes = data.shape[-2]
        elif self.split_by == "circuit":
            n_axes = data.shape[0]
        else:
            n_axes = 1

        def circuit_slice(axis_index):
            if self.split_by == "circuit":
                return slice(axis_index, axis_index + 1)
            return slice(data.shape[0])

        def memslot_slice(axis_index):
            if self.split_by == "memslot":
                return slice(axis_index, axis_index + 1)
            return slice(data.shape[-2])

        if self._n_cols > n_axes:
            n_cols = n_axes
        else:
            n_cols = self._n_cols
        n_rows = int(np.ceil(n_axes / n_cols))
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharex=True,
            sharey=True,
            figsize=(self._axis_width * n_cols, self._axis_height * n_rows),
        )
        axes = np.asarray(axes)

        for i_ax, ax in zip(range(n_axes), axes.flatten()):
            # Remove axis-specific data
            if meas_type == MeasReturnType.AVERAGE:
                index = (circuit_slice(i_ax), memslot_slice(i_ax), slice(0, 2))
            else:
                index = (
                    circuit_slice(i_ax),
                    slice(data.shape[1]),
                    memslot_slice(i_ax),
                    slice(0, 2),
                )

            # Index data for axis
            ax_data = data[index]

            # Generate colours
            def _colour_for_index(*coords) -> int:
                if self.colour_by == "memslot":
                    return coords[-1]
                if self.colour_by == "circuit":
                    return coords[0]
                return 0

            colours = np.fromfunction(_colour_for_index, ax_data.shape[:-1])

            # Reshape ax_data and colours into a list of
            ax_data = np.reshape(ax_data, (-1, 2))
            colours = np.reshape(colours, (-1,))

            # Extract X and Y data for plotting
            if meas_type == MeasReturnType.SINGLE:
                data_x = unp.nominal_values(ax_data[..., 0])
                data_y = unp.nominal_values(ax_data[..., 1])
                x_err = None
                y_err = None
            else:
                # meas_type must be MeasReturnType.AVERAGED
                data_x = unp.nominal_values(ax_data[..., 0])
                data_y = unp.nominal_values(ax_data[..., 1])
                x_err = unp.std_devs(ax_data[..., 0])
                y_err = unp.std_devs(ax_data[..., 1])

            ax.errorbar(
                data_x,
                data_y,
                y_err,
                x_err,
                fmt=" ",
                c="k",
                zorder=10,
            )
            ax.scatter(
                data_x,
                data_y,
                c=colours,
                s=10,
                zorder=11,
            )
            if self.colour_by is not None:
                try:
                    cmap = plt.colorbar()
                    cmap.set_label(f"{self.colour_by} (index)")
                except RuntimeError as e:
                    # If nothing was plotted in this axis, colorbar() will throw an error as no mappable
                    # will exist. But we don't want to throw this error, so capture it and continue.
                    pass
        return fig
