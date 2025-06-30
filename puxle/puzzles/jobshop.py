import chex
import jax
import jax.numpy as jnp

from puxle.utils.annotate import IMG_SIZE
from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import to_uint8, from_uint8

TYPE = jnp.uint16


class JobShop(Puzzle):
    """
    Job Shop Scheduling Problem
    
    Multiple jobs must be processed on multiple machines. Each job has a specific
    sequence of operations on different machines. The goal is to minimize makespan
    (total completion time).
    """
    
    n_jobs: int
    n_machines: int
    
    def define_state_class(self) -> PuzzleState:
        """Defines the state class for JobShop."""
        str_parser = self.get_string_parser()
        
        @state_dataclass
        class State:
            # For each job, which operation is next (0 to n_machines-1)
            job_progress: FieldDescriptor[TYPE, (self.n_jobs,)]
            # For each machine, when it will be available
            machine_available_time: FieldDescriptor[jnp.float32, (self.n_machines,)]
            # For each job, when it will be available
            job_available_time: FieldDescriptor[jnp.float32, (self.n_jobs,)]
            # Current time
            current_time: FieldDescriptor[jnp.float32]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
        
        return State
    
    def define_solve_config_class(self) -> PuzzleState:
        """Defines the solve config class for JobShop."""
        str_parser = self.get_solve_config_string_parser()
        
        @state_dataclass
        class SolveConfig:
            # For each job and operation, which machine to use
            machine_order: FieldDescriptor[TYPE, (self.n_jobs, self.n_machines)]
            # Processing time for each job on each machine
            processing_times: FieldDescriptor[jnp.float32, (self.n_jobs, self.n_machines)]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
        
        return SolveConfig
    
    def __init__(self, n_jobs: int = 3, n_machines: int = 3, **kwargs):
        """
        Initialize the Job Shop puzzle
        
        Args:
            n_jobs: Number of jobs
            n_machines: Number of machines
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        super().__init__(**kwargs)
    
    def get_solve_config_string_parser(self) -> callable:
        def parser(solve_config: "JobShop.SolveConfig", **kwargs):
            return f"Job Shop: {self.n_jobs} jobs, {self.n_machines} machines"
        
        return parser
    
    def get_string_parser(self):
        def parser(state: "JobShop.State", solve_config: "JobShop.SolveConfig" = None, **kwargs):
            result = []
            result.append(f"Current Time: {state.current_time:.1f}")
            result.append("-" * 60)
            
            # Show job progress
            result.append("Job Progress:")
            for j in range(self.n_jobs):
                progress = state.job_progress[j]
                available = state.job_available_time[j]
                status = f"Job {j}: Operation {progress}/{self.n_machines} (available at {available:.1f})"
                result.append(status)
            
            result.append("-" * 60)
            
            # Show machine availability
            result.append("Machine Availability:")
            for m in range(self.n_machines):
                available = state.machine_available_time[m]
                result.append(f"Machine {m}: available at {available:.1f}")
            
            if solve_config is not None:
                result.append("-" * 60)
                result.append("Job Schedule (Machine Order):")
                for j in range(self.n_jobs):
                    machines = [str(solve_config.machine_order[j, op]) for op in range(self.n_machines)]
                    result.append(f"Job {j}: {' -> '.join(machines)}")
            
            return "\n".join(result)
        
        return parser
    
    def get_initial_state(
        self, solve_config: "JobShop.SolveConfig", key=None, data=None
    ) -> "JobShop.State":
        """Start with all jobs at operation 0 and all machines available"""
        return self.State(
            job_progress=jnp.zeros(self.n_jobs, dtype=TYPE),
            machine_available_time=jnp.zeros(self.n_machines, dtype=jnp.float32),
            job_available_time=jnp.zeros(self.n_jobs, dtype=jnp.float32),
            current_time=jnp.float32(0.0)
        )
    
    def get_solve_config(self, key=None, data=None) -> "JobShop.SolveConfig":
        """Generate a random job shop instance"""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        key_order, key_times = jax.random.split(key)
        
        # Generate machine order for each job (random permutation for each job)
        machine_order = jnp.zeros((self.n_jobs, self.n_machines), dtype=TYPE)
        for j in range(self.n_jobs):
            key_order, subkey = jax.random.split(key_order)
            perm = jax.random.permutation(subkey, jnp.arange(self.n_machines, dtype=TYPE))
            machine_order = machine_order.at[j].set(perm)
        
        # Generate processing times (between 1 and 10)
        processing_times = jax.random.uniform(
            key_times, (self.n_jobs, self.n_machines),
            minval=1, maxval=10, dtype=jnp.float32
        )
        
        return self.SolveConfig(
            machine_order=machine_order,
            processing_times=processing_times
        )
    
    def get_neighbours(
        self, solve_config: "JobShop.SolveConfig", state: "JobShop.State", filled: bool = True
    ) -> tuple["JobShop.State", chex.Array]:
        """
        Get neighboring states by scheduling the next operation of each job.
        """
        job_progress = state.job_progress
        machine_avail = state.machine_available_time
        job_avail = state.job_available_time
        current_time = state.current_time
        
        def schedule_job(job_idx):
            # Check if job has more operations
            current_op = job_progress[job_idx]
            has_more_ops = current_op < self.n_machines
            
            # Get machine for current operation
            machine_idx = solve_config.machine_order[job_idx, current_op]
            process_time = solve_config.processing_times[job_idx, current_op]
            
            # Can schedule if job has more operations
            can_schedule = has_more_ops & filled
            
            # Calculate start time (max of job available, machine available, current time)
            start_time = jnp.maximum(
                jnp.maximum(job_avail[job_idx], machine_avail[machine_idx]),
                current_time
            )
            end_time = start_time + process_time
            
            # Update state if valid
            new_job_progress = jnp.where(
                can_schedule,
                job_progress.at[job_idx].set(current_op + 1),
                job_progress
            )
            
            new_machine_avail = jnp.where(
                can_schedule,
                machine_avail.at[machine_idx].set(end_time),
                machine_avail
            )
            
            new_job_avail = jnp.where(
                can_schedule,
                job_avail.at[job_idx].set(end_time),
                job_avail
            )
            
            # Cost is the start time (we want to minimize makespan)
            cost = jnp.where(can_schedule, start_time, jnp.inf)
            
            return self.State(
                job_progress=new_job_progress,
                machine_available_time=new_machine_avail,
                job_available_time=new_job_avail,
                current_time=jnp.where(can_schedule, start_time, current_time)
            ), cost
        
        # Apply to all jobs
        new_states, costs = jax.vmap(schedule_job)(jnp.arange(self.n_jobs))
        
        return new_states, costs
    
    def is_solved(self, solve_config: "JobShop.SolveConfig", state: "JobShop.State") -> bool:
        """Problem is solved when all jobs have completed all operations"""
        return jnp.all(state.job_progress == self.n_machines)
    
    def action_to_string(self, action: int) -> str:
        """Return string representation of the action"""
        return f"Schedule next operation of job {action}"
    
    def get_img_parser(self):
        """Generate Gantt chart representation of the schedule"""
        import cv2
        import numpy as np
        
        def img_func(state: "JobShop.State", solve_config: "JobShop.SolveConfig" = None, **kwargs):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255
            
            if solve_config is None:
                return img
            
            # Calculate makespan for scaling
            max_time = max(
                np.max(state.machine_available_time),
                np.max(state.job_available_time),
                20.0  # Minimum scale
            )
            
            # Colors for jobs
            job_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]
            
            # Gantt chart parameters
            margin = 40
            chart_width = IMG_SIZE[0] - 2 * margin
            chart_height = IMG_SIZE[1] - 2 * margin
            machine_height = chart_height // (self.n_machines + 1)
            
            # Draw time axis
            cv2.line(img, (margin, IMG_SIZE[1] - margin), 
                     (IMG_SIZE[0] - margin, IMG_SIZE[1] - margin), (0, 0, 0), 2)
            
            # Draw time labels
            for t in range(0, int(max_time) + 1, 5):
                x = margin + int(t * chart_width / max_time)
                cv2.line(img, (x, IMG_SIZE[1] - margin - 5), 
                         (x, IMG_SIZE[1] - margin + 5), (0, 0, 0), 1)
                cv2.putText(img, str(t), (x - 10, IMG_SIZE[1] - margin + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Draw machine lanes
            for m in range(self.n_machines):
                y = margin + m * machine_height + machine_height // 2
                cv2.line(img, (margin, y), (IMG_SIZE[0] - margin, y), (200, 200, 200), 1)
                cv2.putText(img, f"M{m}", (10, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Track scheduled operations (simplified visualization)
            # In a real implementation, we'd need to track the full schedule
            for j in range(self.n_jobs):
                if state.job_progress[j] > 0:
                    # Draw a simple bar for completed operations
                    color = job_colors[j % len(job_colors)]
                    
                    # For simplicity, just show progress
                    progress_ratio = state.job_progress[j] / self.n_machines
                    bar_width = int(progress_ratio * chart_width * 0.8)
                    
                    y = margin + j * 20 + 10
                    cv2.rectangle(img, (margin, y), (margin + bar_width, y + 15), color, -1)
                    cv2.putText(img, f"Job {j}", (margin + bar_width + 5, y + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # Draw current time line
            if state.current_time > 0:
                x = margin + int(state.current_time * chart_width / max_time)
                cv2.line(img, (x, margin), (x, IMG_SIZE[1] - margin), (255, 0, 0), 1)
            
            # Status text
            completed = int(jnp.sum(state.job_progress == self.n_machines))
            status = f"Completed: {completed}/{self.n_jobs} jobs"
            cv2.putText(img, status, (margin, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return img
        
        return img_func
    
    @property
    def fixed_target(self) -> bool:
        """Job Shop has a fixed target (all jobs completed)"""
        return True
    
    def get_solve_config_img_parser(self):
        """Generate image representation of the solve config"""
        import cv2
        import numpy as np
        
        def img_func(solve_config: "JobShop.SolveConfig", **kwargs):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255
            
            machine_order = np.array(solve_config.machine_order)
            processing_times = np.array(solve_config.processing_times)
            
            # Title
            title = f"Job Shop: {machine_order.shape[0]} jobs, {machine_order.shape[1]} machines"
            cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw job information
            start_y = 50
            row_height = 30
            
            # Headers
            cv2.putText(img, "Job", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(img, "Machine Order", (60, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(img, "Processing Times", (250, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Job colors
            job_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
            ]
            
            for j in range(machine_order.shape[0]):
                y = start_y + (j + 1) * row_height
                color = job_colors[j % len(job_colors)]
                
                # Job number
                cv2.putText(img, f"{j}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Machine order
                order_str = " -> ".join([f"M{m}" for m in machine_order[j]])
                cv2.putText(img, order_str, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                
                # Processing times
                times_str = ", ".join([f"{t:.1f}" for t in processing_times[j]])
                cv2.putText(img, times_str, (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # Total processing time per job
            y = start_y + (machine_order.shape[0] + 2) * row_height
            cv2.putText(img, "Total time per job:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            for j in range(machine_order.shape[0]):
                total_time = np.sum(processing_times[j])
                color = job_colors[j % len(job_colors)]
                cv2.putText(img, f"Job {j}: {total_time:.1f}", (150 + j * 80, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return img
        
        return img_func