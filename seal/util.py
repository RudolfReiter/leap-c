import atexit
from pathlib import Path
import shutil
from tempfile import mkdtemp

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import casadi as ca
import torch


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


class AcadosFileManager:
    """A simple class to manage the export directory for acados solvers."""

    def __init__(
        self, export_directory: Path | None = None, cleanup: bool = True,
    ):
        """Initialize the export directory manager.

        Args:
            export_directory: The export directory if None create a folder in /tmp.
            cleanup: Whether to delete the export directory on exit or when the 
                instance of the AcadosFileManager is garbage collected.
        """
        self.export_directory = (
            Path(mkdtemp()) if export_directory is None else export_directory
        )

        self.cleanup = cleanup
        if cleanup:
            atexit.register(self.__del__)

    def setup_acados_ocp_solver(self, ocp: AcadosOcp) -> AcadosOcpSolver:
        """Setup an acados ocp solver with path management.

        We set the json file and the code export directory.

        Args:
            ocp: The acados ocp object.

        Returns:
            AcadosOcpSolver: The acados ocp solver.
        """
        ocp.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosOcpSolver(ocp, json_file=json_file)

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def setup_acados_sim_solver(self, sim: AcadosSim) -> AcadosSimSolver:
        """Setup an acados sim solver with path management.

        We set the json file and the code export directory.

        Args:
            sim: The acados sim object.

        Returns:
            AcadosSimSolver: The acados sim solver.
        """
        sim.code_export_directory = str(self.export_directory / "c_generated_code")
        json_file = str(self.export_directory / "acados_ocp.json")

        solver = AcadosSimSolver(sim, json_file=json_file)

        # we add the acados file manager to the solver to ensure
        # the export directory is deleted when the solver is garbage collected
        solver.__acados_file_manager = self  # type: ignore

        return solver

    def __del__(self):
        if self.cleanup:
            shutil.rmtree(self.export_directory, ignore_errors=True)