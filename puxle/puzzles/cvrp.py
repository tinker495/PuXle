import chex
import jax
import jax.numpy as jnp

from puxle.utils.annotate import IMG_SIZE
from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import to_uint8, from_uint8

TYPE = jnp.uint16  # Support more than 255 customers


class CVRP(Puzzle):
    """
    Capacitated Vehicle Routing Problem
    
    Multiple vehicles with limited capacity must visit all customers exactly once,
    starting and ending at a depot. The goal is to minimize total distance traveled.
    """
    
    n_customers: int
    n_vehicles: int
    vehicle_capacity: int
    
    def define_state_class(self) -> PuzzleState:
        """Defines the state class for CVRP."""
        str_parser = self.get_string_parser()
        visited_mask = jnp.zeros(self.n_customers + 1, dtype=jnp.bool_)  # +1 for depot
        packed_mask = to_uint8(visited_mask)
        n_customers = self.n_customers
        
        @state_dataclass
        class State:
            visited: FieldDescriptor[jnp.uint8, packed_mask.shape, packed_mask]
            current_location: FieldDescriptor[TYPE]
            current_vehicle: FieldDescriptor[TYPE]
            remaining_capacity: FieldDescriptor[jnp.float32]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
            
            @property
            def packed(self):
                return State(
                    visited=to_uint8(self.visited),
                    current_location=self.current_location,
                    current_vehicle=self.current_vehicle,
                    remaining_capacity=self.remaining_capacity
                )
            
            @property
            def unpacked(self):
                return State(
                    visited=from_uint8(self.visited, (n_customers + 1,)),
                    current_location=self.current_location,
                    current_vehicle=self.current_vehicle,
                    remaining_capacity=self.remaining_capacity
                )
        
        return State
    
    def define_solve_config_class(self) -> PuzzleState:
        """Defines the solve config class for CVRP."""
        str_parser = self.get_solve_config_string_parser()
        
        @state_dataclass
        class SolveConfig:
            locations: FieldDescriptor[jnp.float32, (self.n_customers + 1, 2)]  # +1 for depot
            demands: FieldDescriptor[jnp.float32, (self.n_customers + 1,)]  # depot has 0 demand
            distance_matrix: FieldDescriptor[jnp.float32, (self.n_customers + 1, self.n_customers + 1)]
            vehicle_capacity: FieldDescriptor[jnp.float32]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
        
        return SolveConfig
    
    def __init__(self, size: int = 20, n_vehicles: int = 4, vehicle_capacity: int = 150, **kwargs):
        """
        Initialize the CVRP puzzle
        
        Args:
            n_customers: Number of customers to visit
            n_vehicles: Number of available vehicles
            vehicle_capacity: Capacity of each vehicle
        """
        self.n_customers = size
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity
        super().__init__(**kwargs)
    
    def get_solve_config_string_parser(self) -> callable:
        def parser(solve_config: "CVRP.SolveConfig", **kwargs):
            return f"CVRP: {self.n_customers} customers, {self.n_vehicles} vehicles, capacity: {solve_config.vehicle_capacity}"
        
        return parser
    
    def get_string_parser(self):
        def parser(state: "CVRP.State", solve_config: "CVRP.SolveConfig" = None, **kwargs):
            visited = state.unpacked.visited
            
            if solve_config is None:
                return f"CVRP State: Vehicle {state.current_vehicle} at location {state.current_location}"
            
            result = []
            result.append(f"Vehicle {state.current_vehicle}/{self.n_vehicles} at location {state.current_location}")
            result.append(f"Remaining capacity: {state.remaining_capacity:.1f}/{solve_config.vehicle_capacity:.1f}")
            result.append("-" * 50)
            
            # Show visited status
            visited_customers = jnp.sum(visited[1:])  # Exclude depot
            result.append(f"Visited: {visited_customers}/{self.n_customers} customers")
            
            # Show customer details
            result.append("-" * 50)
            result.append("Customer | Demand | Visited")
            result.append("-" * 50)
            
            for i in range(1, self.n_customers + 1):
                visited_str = "✓" if visited[i] else " "
                result.append(f"{i:8d} | {solve_config.demands[i]:6.1f} | {visited_str:^7}")
            
            return "\n".join(result)
        
        return parser
    
    def get_initial_state(
        self, solve_config: "CVRP.SolveConfig", key=None, data=None
    ) -> "CVRP.State":
        """Start at depot with first vehicle"""
        visited = jnp.zeros(self.n_customers + 1, dtype=jnp.bool_)
        visited = visited.at[0].set(True)  # Mark depot as visited
        
        return self.State(
            visited=visited,
            current_location=jnp.array(0, dtype=TYPE),  # Start at depot
            current_vehicle=jnp.array(0, dtype=TYPE),
            remaining_capacity=solve_config.vehicle_capacity
        ).packed
    
    def get_solve_config(self, key=None, data=None) -> "CVRP.SolveConfig":
        """Generate random customer locations and demands"""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        key_loc, key_demand = jax.random.split(key)
        
        # Generate locations (depot at origin)
        depot_location = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
        customer_locations = jax.random.uniform(
            key_loc, (self.n_customers, 2), minval=0, maxval=1, dtype=jnp.float32
        )
        locations = jnp.concatenate([depot_location, customer_locations], axis=0)
        
        # Generate demands (depot has 0 demand)
        depot_demand = jnp.array([0.0], dtype=jnp.float32)
        customer_demands = jax.random.uniform(
            key_demand, (self.n_customers,), 
            minval=15, maxval=60, dtype=jnp.float32
        )
        demands = jnp.concatenate([depot_demand, customer_demands])
        
        # Calculate distance matrix
        distance_matrix = jnp.linalg.norm(
            locations[:, None] - locations[None, :], axis=-1
        ).astype(jnp.float32)
        
        return self.SolveConfig(
            locations=locations,
            demands=demands,
            distance_matrix=distance_matrix,
            vehicle_capacity=jnp.float32(self.vehicle_capacity)
        )
    
    def get_neighbours(
        self, solve_config: "CVRP.SolveConfig", state: "CVRP.State", filled: bool = True
    ) -> tuple["CVRP.State", chex.Array]:
        """
        Get neighboring states by:
        1. Visiting an unvisited customer (if capacity allows)
        2. Returning to depot to switch vehicles
        """
        visited = state.unpacked.visited
        current_loc = state.current_location
        current_vehicle = state.current_vehicle
        remaining_cap = state.remaining_capacity
        
        # Total locations = n_customers + 1 (depot)
        n_locations = self.n_customers + 1
        
        def move_to_location(next_loc):
            is_depot = next_loc == 0
            is_visited = visited[next_loc]
            demand = solve_config.demands[next_loc]
            
            # Can visit if:
            # 1. It's the depot (always can return)
            # 2. It's an unvisited customer with demand <= remaining capacity
            can_visit_customer = ~is_visited & (demand <= remaining_cap) & ~is_depot
            can_visit_depot = is_depot & (current_loc != 0)  # Can't stay at depot
            can_visit = can_visit_customer | can_visit_depot
            
            # Update state if move is valid
            new_visited = jnp.where(can_visit & ~is_depot, visited.at[next_loc].set(True), visited)
            
            # If returning to depot, switch to next vehicle
            new_vehicle = jnp.where(
                is_depot & can_visit,
                jnp.minimum(current_vehicle + 1, self.n_vehicles - 1),
                current_vehicle
            )
            
            # Update capacity
            new_capacity = jnp.where(
                is_depot & can_visit,
                solve_config.vehicle_capacity,  # Reset capacity
                remaining_cap - demand  # Reduce by demand
            )
            
            # Calculate cost (distance)
            distance = solve_config.distance_matrix[current_loc, next_loc]
            cost = jnp.where(can_visit & filled, distance, jnp.inf)
            
            # Only update location if valid
            final_location = jnp.where(can_visit & filled, next_loc, current_loc)
            final_visited = jnp.where(can_visit & filled, new_visited, visited)
            final_vehicle = jnp.where(can_visit & filled, new_vehicle, current_vehicle)
            final_capacity = jnp.where(can_visit & filled, new_capacity, remaining_cap)
            
            return self.State(
                visited=final_visited,
                current_location=final_location,
                current_vehicle=final_vehicle,
                remaining_capacity=final_capacity
            ).packed, cost
        
        # Apply move to all locations
        new_states, costs = jax.vmap(move_to_location)(jnp.arange(n_locations, dtype=TYPE))
        
        return new_states, costs
    
    def is_solved(self, solve_config: "CVRP.SolveConfig", state: "CVRP.State") -> bool:
        """Problem is solved when all customers are visited and we're back at depot"""
        visited = state.unpacked.visited
        all_customers_visited = jnp.all(visited[1:])  # Exclude depot
        at_depot = state.current_location == 0
        return all_customers_visited & at_depot
    
    def action_to_string(self, action: int) -> str:
        """Return string representation of the action"""
        if action == 0:
            return "Return to depot"
        else:
            return f"Visit customer {action}"
    
    def get_img_parser(self):
        """Generate image representation of the CVRP state"""
        import cv2
        import numpy as np
        
        def img_func(
            state: "CVRP.State", 
            path: list["CVRP.State"] = None,
            idx: int = None,
            solve_config: "CVRP.SolveConfig" = None, 
            **kwargs
        ):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255
            
            if solve_config is None:
                return img
            
            visited = state.unpacked.visited
            locations = np.array(solve_config.locations)
            
            # Scale locations to image coordinates
            margin = 50
            scale = IMG_SIZE[0] - 2 * margin
            scaled_locations = margin + locations * scale
            
            # Vehicle colors (more distinct colors)
            vehicle_colors = [
                (255, 0, 0),    # Red
                (0, 200, 0),    # Green  
                (0, 0, 255),    # Blue
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (128, 0, 128),  # Purple
                (255, 192, 203) # Pink
            ]
            
            # Draw paths if available
            if path and idx is not None and len(path) > 1:
                # Track vehicle routes by analyzing the path
                vehicle_routes = {v: [] for v in range(self.n_vehicles)}
                current_vehicle = 0
                
                for i in range(min(idx + 1, len(path))):
                    route_state = path[i]
                    location = int(route_state.current_location)
                    vehicle = int(route_state.current_vehicle)
                    
                    # If vehicle changed, we're at depot
                    if vehicle != current_vehicle:
                        current_vehicle = vehicle
                        vehicle_routes[vehicle] = [0]  # Start new route from depot
                    
                    if len(vehicle_routes[vehicle]) == 0 or vehicle_routes[vehicle][-1] != location:
                        vehicle_routes[vehicle].append(location)
                
                # Draw routes for each vehicle
                for vehicle_id, route in vehicle_routes.items():
                    if len(route) > 1:
                        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
                        route_points = []
                        for loc in route:
                            x, y = int(scaled_locations[loc, 0]), int(scaled_locations[loc, 1])
                            route_points.append((x, y))
                        
                        # Draw route lines
                        for i in range(len(route_points) - 1):
                            cv2.line(img, route_points[i], route_points[i+1], color, 3)
                        
                        # Draw direction arrows
                        for i in range(len(route_points) - 1):
                            start = np.array(route_points[i])
                            end = np.array(route_points[i+1])
                            direction = end - start
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                arrow_pos = start + 0.7 * (end - start)
                                arrow_end = arrow_pos + 8 * direction
                                cv2.arrowedLine(img, tuple(arrow_pos.astype(int)), 
                                              tuple(arrow_end.astype(int)), color, 2, tipLength=0.3)
            
            # Draw all locations
            for i in range(len(locations)):
                x, y = int(scaled_locations[i, 0]), int(scaled_locations[i, 1])
                
                if i == 0:  # Depot
                    cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (0, 0, 0), -1)
                    cv2.putText(img, "DEPOT", (x-20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                else:  # Customer
                    # Color based on visited status
                    if visited[i]:
                        color = (100, 255, 100)  # Light green for visited
                        cv2.circle(img, (x, y), 8, color, -1)
                        cv2.circle(img, (x, y), 8, (0, 150, 0), 2)  # Dark green border
                    else:
                        color = (200, 200, 200)  # Gray for unvisited
                        cv2.circle(img, (x, y), 8, color, -1)
                        cv2.circle(img, (x, y), 8, (100, 100, 100), 2)  # Dark gray border
                    
                    # Customer number
                    cv2.putText(img, str(i), (x-5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw demand for customers
                if i > 0:
                    demand_text = f"d:{solve_config.demands[i]:.0f}"
                    cv2.putText(img, demand_text, (x+12, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            
            # Highlight current location with larger circle
            curr_x = int(scaled_locations[state.current_location, 0])
            curr_y = int(scaled_locations[state.current_location, 1])
            current_color = vehicle_colors[int(state.current_vehicle) % len(vehicle_colors)]
            cv2.circle(img, (curr_x, curr_y), 12, current_color, 3)
            
            # Draw comprehensive vehicle info panel
            panel_x, panel_y = 10, 20
            panel_width = 280
            panel_height = 25 * self.n_vehicles + 60
            
            # Draw panel background
            cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (240, 240, 240), -1)
            cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), 2)
            
            # Vehicle info header
            cv2.putText(img, "VEHICLES STATUS", (panel_x + 10, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw each vehicle's status
            for v in range(self.n_vehicles):
                color = vehicle_colors[v % len(vehicle_colors)]
                y_pos = panel_y + 45 + v * 25
                
                # Vehicle indicator
                cv2.rectangle(img, (panel_x + 10, y_pos - 8), (panel_x + 25, y_pos + 8), color, -1)
                cv2.putText(img, str(v+1), (panel_x + 15, y_pos + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Vehicle info
                if v == state.current_vehicle:
                    # Current vehicle - show actual capacity
                    used_capacity = solve_config.vehicle_capacity - state.remaining_capacity
                    capacity_text = f"Vehicle {v+1}: {used_capacity:.0f}/{solve_config.vehicle_capacity:.0f} ACTIVE"
                    cv2.putText(img, capacity_text, (panel_x + 35, y_pos + 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                    
                    # Capacity bar
                    bar_x = panel_x + 35
                    bar_y = y_pos + 8
                    bar_width = 150
                    bar_height = 6
                    
                    # Background bar
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                 (200, 200, 200), -1)
                    
                    # Filled bar
                    fill_ratio = used_capacity / solve_config.vehicle_capacity
                    fill_width = int(bar_width * fill_ratio)
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                                 color, -1)
                    
                    # Capacity percentage
                    percent = fill_ratio * 100
                    cv2.putText(img, f"{percent:.1f}%", (bar_x + bar_width + 5, bar_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                else:
                    # Inactive vehicle
                    cv2.putText(img, f"Vehicle {v+1}: Standby", (panel_x + 35, y_pos + 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # Draw progress info at bottom
            n_visited = int(jnp.sum(visited[1:]))
            total_demand = float(jnp.sum(solve_config.demands[1:]))
            served_demand = float(jnp.sum(solve_config.demands[1:] * visited[1:]))
            
            progress_y = IMG_SIZE[1] - 40
            progress_text = f"Progress: {n_visited}/{self.n_customers} customers"
            cv2.putText(img, progress_text, (10, progress_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            demand_text = f"Demand served: {served_demand:.0f}/{total_demand:.0f} ({served_demand/total_demand*100:.1f}%)"
            cv2.putText(img, demand_text, (10, progress_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return img
        
        return img_func
    
    @property
    def fixed_target(self) -> bool:
        """CVRP has a fixed target (all customers visited)"""
        return True
    
    def get_solve_config_img_parser(self):
        """Generate image representation of the solve config"""
        import cv2
        import numpy as np
        
        def img_func(solve_config: "CVRP.SolveConfig", **kwargs):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 255
            
            locations = np.array(solve_config.locations)
            demands = np.array(solve_config.demands)
            
            # Scale locations to image coordinates
            margin = 40
            scale = IMG_SIZE[0] - 2 * margin
            scaled_locations = margin + locations * scale
            
            # Draw all locations
            for i in range(len(locations)):
                x, y = int(scaled_locations[i, 0]), int(scaled_locations[i, 1])
                
                if i == 0:  # Depot
                    cv2.rectangle(img, (x-8, y-8), (x+8, y+8), (0, 0, 0), -1)
                    cv2.putText(img, "DEPOT", (x-20, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                else:  # Customer
                    cv2.circle(img, (x, y), 6, (150, 150, 150), -1)
                    cv2.putText(img, str(i), (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    cv2.putText(img, f"d:{demands[i]:.0f}", (x+8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            
            # Draw title
            title = f"CVRP: {len(locations)-1} customers, Vehicle capacity: {solve_config.vehicle_capacity:.0f}"
            cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw total demand
            total_demand = np.sum(demands[1:])
            info = f"Total demand: {total_demand:.0f}"
            cv2.putText(img, info, (10, IMG_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return img
        
        return img_func