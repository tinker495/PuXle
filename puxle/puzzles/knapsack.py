import chex
import jax
import jax.numpy as jnp

from puxle.utils.annotate import IMG_SIZE
from puxle.core.puzzle_base import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils.util import to_uint8, from_uint8

TYPE = jnp.uint8


class Knapsack(Puzzle):
    """
    0-1 Knapsack Problem
    
    Given a set of items with weights and values, and a knapsack with limited capacity,
    determine which items to include in the knapsack to maximize the total value
    without exceeding the weight capacity.
    """
    
    n_items: int
    capacity: int
    
    def define_state_class(self) -> PuzzleState:
        """Defines the state class for Knapsack using xtructure."""
        str_parser = self.get_string_parser()
        mask = jnp.zeros(self.n_items, dtype=jnp.bool_)
        packed_mask = to_uint8(mask)
        n_items = self.n_items
        
        @state_dataclass
        class State:
            selected: FieldDescriptor[TYPE, packed_mask.shape, packed_mask]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
            
            @property
            def packed(self):
                return State(selected=to_uint8(self.selected))
            
            @property
            def unpacked(self):
                return State(selected=from_uint8(self.selected, (n_items,)))
        
        return State
    
    def define_solve_config_class(self) -> PuzzleState:
        """Defines the solve config class for Knapsack."""
        str_parser = self.get_solve_config_string_parser()
        
        @state_dataclass
        class SolveConfig:
            weights: FieldDescriptor[jnp.float32, (self.n_items,)]
            values: FieldDescriptor[jnp.float32, (self.n_items,)]
            capacity: FieldDescriptor[jnp.float32]
            
            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)
        
        return SolveConfig
    
    def __init__(self, size: int = 20, capacity: int = 50, **kwargs):
        """
        Initialize the Knapsack puzzle
        
        Args:
            size: Number of items
            capacity: Weight capacity of the knapsack
        """
        self.n_items = size
        self.capacity = capacity
        super().__init__(**kwargs)
    
    def get_solve_config_string_parser(self) -> callable:
        def parser(solve_config: "Knapsack.SolveConfig", **kwargs):
            return f"Knapsack: {self.n_items} items, capacity: {solve_config.capacity}"
        
        return parser
    
    def get_string_parser(self):
        def parser(state: "Knapsack.State", solve_config: "Knapsack.SolveConfig" = None, **kwargs):
            selected = state.unpacked.selected
            
            if solve_config is None:
                return f"Knapsack State: {jnp.sum(selected)} items selected"
            
            total_weight = jnp.sum(selected * solve_config.weights)
            total_value = jnp.sum(selected * solve_config.values)
            
            result = []
            result.append(f"Capacity: {solve_config.capacity:.1f} | Used: {total_weight:.1f} | Value: {total_value:.1f}")
            result.append("-" * 60)
            result.append("Item | Weight | Value | Selected")
            result.append("-" * 60)
            
            for i in range(self.n_items):
                selected_str = "✓" if selected[i] else " "
                result.append(f"{i:4d} | {solve_config.weights[i]:6.1f} | {solve_config.values[i]:5.1f} | {selected_str:^8}")
            
            return "\n".join(result)
        
        return parser
    
    def get_initial_state(
        self, solve_config: "Knapsack.SolveConfig", key=None, data=None
    ) -> "Knapsack.State":
        """Start with no items selected"""
        selected = jnp.zeros(self.n_items, dtype=jnp.bool_)
        return self.State(selected=selected).packed
    
    def get_solve_config(self, key=None, data=None) -> "Knapsack.SolveConfig":
        """Generate random weights and values for items"""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        key_weights, key_values = jax.random.split(key)
        
        # Generate weights between 1 and 20
        weights = jax.random.uniform(key_weights, (self.n_items,), minval=1, maxval=20, dtype=jnp.float32)
        
        # Generate values between 10 and 100
        values = jax.random.uniform(key_values, (self.n_items,), minval=10, maxval=100, dtype=jnp.float32)
        
        return self.SolveConfig(
            weights=weights,
            values=values,
            capacity=jnp.float32(self.capacity)
        )
    
    def get_neighbours(
        self, solve_config: "Knapsack.SolveConfig", state: "Knapsack.State", filled: bool = True
    ) -> tuple["Knapsack.State", chex.Array]:
        """
        Get neighboring states by toggling each item's selection.
        Cost is negative value if adding item, positive value if removing.
        """
        selected = state.unpacked.selected
        
        def toggle_item(idx):
            new_selected = selected.at[idx].set(~selected[idx])
            
            # Calculate total weight with new selection
            total_weight = jnp.sum(new_selected * solve_config.weights)
            
            # Check if weight constraint is satisfied
            valid = total_weight <= solve_config.capacity
            
            # Cost: negative value for adding (we want to maximize), positive for removing
            # Use infinity for invalid moves
            value_change = solve_config.values[idx] * jnp.where(selected[idx], 1, -1)
            cost = jnp.where(valid & filled, value_change, jnp.inf)
            
            # Only update state if valid
            final_selected = jnp.where(valid & filled, new_selected, selected)
            
            return self.State(selected=final_selected).packed, cost
        
        # Apply toggle to all items
        new_states, costs = jax.vmap(toggle_item)(jnp.arange(self.n_items))
        
        return new_states, costs
    
    def is_solved(self, solve_config: "Knapsack.SolveConfig", state: "Knapsack.State") -> bool:
        """
        In knapsack, there's no specific target state.
        We could define "solved" as having selected at least one item.
        """
        return jnp.sum(state.unpacked.selected) > 0
    
    def action_to_string(self, action: int) -> str:
        """Return string representation of the action"""
        return f"Toggle item {action}"
    
    def get_img_parser(self):
        """Generate image representation of the knapsack state"""
        import cv2
        import numpy as np
        
        def img_func(state: "Knapsack.State", solve_config: "Knapsack.SolveConfig" = None, **kwargs):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 240
            
            if solve_config is None:
                return img
            
            selected = state.unpacked.selected
            
            # Calculate metrics
            total_weight = jnp.sum(selected * solve_config.weights)
            total_value = jnp.sum(selected * solve_config.values)
            capacity_ratio = total_weight / solve_config.capacity
            
            # Draw capacity bar
            bar_height = 30
            bar_width = IMG_SIZE[0] - 40
            bar_x = 20
            bar_y = 20
            
            # Background bar
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
            
            # Filled portion
            fill_width = int(bar_width * min(capacity_ratio, 1.0))
            fill_color = (0, 255, 0) if capacity_ratio <= 1.0 else (0, 0, 255)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
            
            # Text
            text = f"Weight: {total_weight:.1f}/{solve_config.capacity:.1f}"
            cv2.putText(img, text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            text = f"Value: {total_value:.1f}"
            cv2.putText(img, text, (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw items
            items_start_y = 80
            item_height = (IMG_SIZE[1] - items_start_y - 20) // self.n_items
            item_height = min(item_height, 20)
            
            for i in range(self.n_items):
                y = items_start_y + i * item_height
                
                # Item bar (weight)
                weight_ratio = solve_config.weights[i] / jnp.max(solve_config.weights)
                weight_width = int(bar_width * 0.4 * weight_ratio)
                
                # Value bar
                value_ratio = solve_config.values[i] / jnp.max(solve_config.values)
                value_width = int(bar_width * 0.4 * value_ratio)
                
                # Background
                item_color = (150, 255, 150) if selected[i] else (220, 220, 220)
                cv2.rectangle(img, (bar_x, y), (bar_x + bar_width, y + item_height - 2), item_color, -1)
                
                # Weight bar (blue)
                cv2.rectangle(img, (bar_x, y), (bar_x + weight_width, y + item_height - 2), (255, 150, 0), -1)
                
                # Value bar (green) - offset to the right
                value_x = bar_x + bar_width // 2
                cv2.rectangle(img, (value_x, y), (value_x + value_width, y + item_height - 2), (0, 200, 0), -1)
                
                # Item number
                text = f"{i}"
                cv2.putText(img, text, (5, y + item_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            return img
        
        return img_func
    
    @property
    def fixed_target(self) -> bool:
        """Knapsack doesn't have a fixed target state"""
        return False
    
    @property  
    def has_target(self) -> bool:
        """Knapsack doesn't have a specific target state"""
        return False
    
    def get_solve_config_img_parser(self):
        """Generate image representation of the solve config"""
        import cv2
        import numpy as np
        
        def img_func(solve_config: "Knapsack.SolveConfig", **kwargs):
            img = np.ones(IMG_SIZE + (3,), np.uint8) * 240
            
            # Draw item information
            n_items = len(solve_config.weights)
            item_height = IMG_SIZE[1] // (n_items + 2)
            item_height = min(item_height, 30)
            
            # Title
            cv2.putText(img, f"Knapsack Problem - Capacity: {solve_config.capacity:.1f}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            start_y = 40
            max_weight = np.max(solve_config.weights)
            max_value = np.max(solve_config.values)
            
            for i in range(n_items):
                y = start_y + i * item_height
                
                # Draw weight bar (blue)
                weight_ratio = solve_config.weights[i] / max_weight
                weight_width = int((IMG_SIZE[0] - 60) * 0.4 * weight_ratio)
                cv2.rectangle(img, (40, y), (40 + weight_width, y + item_height - 2), 
                             (255, 150, 0), -1)
                
                # Draw value bar (green)
                value_ratio = solve_config.values[i] / max_value
                value_width = int((IMG_SIZE[0] - 60) * 0.4 * value_ratio)
                value_x = IMG_SIZE[0] // 2
                cv2.rectangle(img, (value_x, y), (value_x + value_width, y + item_height - 2), 
                             (0, 200, 0), -1)
                
                # Labels
                cv2.putText(img, f"Item {i}", (5, y + item_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                cv2.putText(img, f"W:{solve_config.weights[i]:.1f}", 
                           (45 + weight_width, y + item_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                cv2.putText(img, f"V:{solve_config.values[i]:.1f}", 
                           (value_x + value_width + 5, y + item_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            return img
        
        return img_func 