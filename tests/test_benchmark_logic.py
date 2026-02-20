import pytest
import jax.numpy as jnp
from puxle.benchmark.benchmark import Benchmark, BenchmarkSample
from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, state_dataclass

@state_dataclass(bitpack="off")
class MockState:
    val: FieldDescriptor.scalar(dtype=jnp.int32)

@state_dataclass(bitpack="off")
class MockSolveConfig:
    TargetState: FieldDescriptor.scalar(dtype=MockState)

class MockPuzzle(Puzzle):
    def define_state_class(self):
        return MockState

    def define_solve_config_class(self):
        return MockSolveConfig

    def __init__(self, **kwargs):
        self.action_size = 2
        super().__init__(**kwargs)

    def get_initial_state(self, solve_config, key=None, data=None):
        return MockState(val=0)

    def get_solve_config(self, key=None, data=None):
        return MockSolveConfig(TargetState=MockState(val=10))

    def get_actions(self, solve_config, state, action, filled=True):
        # Action 0 adds 1, action 1 adds 2
        add_val = jnp.where(action == 0, 1, 2)
        return MockState(val=state.val + add_val), jnp.where(filled, 1.0, jnp.inf)

    def is_solved(self, solve_config, state):
        return state.val == solve_config.TargetState.val

    def get_string_parser(self):
        return lambda s, **kwargs: str(s.val)

    def get_img_parser(self):
        return lambda s, **kwargs: jnp.zeros((10, 10, 3))

    def action_to_string(self, action):
        return "add1" if action == 0 else "add2"

class MockBenchmark(Benchmark):
    def build_puzzle(self):
        return MockPuzzle()

    def load_dataset(self):
        return {"test1": BenchmarkSample(
            state=MockState(val=0),
            solve_config=MockSolveConfig(TargetState=MockState(val=10)),
            optimal_action_sequence=["add2"] * 5,
            optimal_path=None,
            optimal_path_costs=5.0
        )}

    def sample_ids(self):
        return ["test1"]

    def get_sample(self, sample_id):
        return self.dataset[sample_id]

def test_verify_solution_valid_optimal():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    # Exact optimal
    assert bm.verify_solution(sample, action_sequence=["add2"] * 5) is True

def test_verify_solution_valid_suboptimal():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    # Suboptimal taking 10 steps (optimal is 5)
    assert bm.verify_solution(sample, action_sequence=["add1"] * 10) is False

def test_verify_solution_invalid_unsolved():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    # Only adds to 8, not 10
    assert bm.verify_solution(sample, action_sequence=["add2"] * 4) is False

def test_verify_solution_unknown_action():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    with pytest.raises(KeyError):
        bm.verify_solution(sample, action_sequence=["unknown"])

def test_verify_solution_with_states_instead_of_actions():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    # Construct sequence of states
    states = [MockState(val=i*2) for i in range(6)] # 0, 2, 4, 6, 8, 10
    assert bm.verify_solution(sample, states=states) is True

def test_verify_solution_no_optimal_info():
    bm = MockBenchmark()
    sample = bm.get_sample("test1")
    # Create sample without optimal info
    sample_no_opt = BenchmarkSample(
        state=sample.state,
        solve_config=sample.solve_config,
        optimal_action_sequence=None,
        optimal_path=None,
        optimal_path_costs=None
    )
    # Since there's no optimal sequence, it just checks validity and returns None
    assert bm.verify_solution(sample_no_opt, action_sequence=["add2"] * 5) is None
