from pathlib import Path

import pytest

from puxle.pddls.pddl import PDDL


class TestPDDLErrors:
    """Error handling tests."""

    def test_malformed_pddl_error(self):
        """Test that malformed PDDL parse failures are wrapped as ValueError."""
        bad_domain = str(
            (Path(__file__).resolve().parents[1] / "pddl_data" / "bad" / "domain.pddl")
        )
        good_problem = str(
            (
                Path(__file__).resolve().parents[1]
                / "pddl_data"
                / "simple_move"
                / "problem.pddl"
            )
        )

        with pytest.raises(ValueError, match="Failed to parse PDDL domain file"):
            PDDL(bad_domain, good_problem)

    def test_missing_file_error(self):
        """Test that missing files raise appropriate exceptions."""
        non_existent_domain = "/non/existent/domain.pddl"
        non_existent_problem = "/non/existent/problem.pddl"

        with pytest.raises(
            ValueError
        ):  # PDDL class wraps FileNotFoundError in ValueError
            PDDL(non_existent_domain, non_existent_problem)

        # Test with one existing and one missing file
        good_domain = str(
            (
                Path(__file__).resolve().parents[1]
                / "pddl_data"
                / "simple_move"
                / "domain.pddl"
            )
        )
        with pytest.raises(
            ValueError
        ):  # PDDL class wraps FileNotFoundError in ValueError
            PDDL(good_domain, non_existent_problem)
