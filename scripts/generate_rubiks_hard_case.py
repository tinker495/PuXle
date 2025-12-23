
import jax
import jax.numpy as jnp
import numpy as np
from xtructure import FieldDescriptor

# Monkey-patch FieldDescriptor
def tensor(dtype, shape, fill_value=None):
    return FieldDescriptor(dtype=dtype, intrinsic_shape=shape, fill_value=fill_value)

def scalar(dtype, fill_value=None):
    return FieldDescriptor(dtype=dtype, intrinsic_shape=(), fill_value=fill_value)

if not hasattr(FieldDescriptor, 'tensor'):
    FieldDescriptor.tensor = staticmethod(tensor)

if not hasattr(FieldDescriptor, 'scalar'):
    FieldDescriptor.scalar = staticmethod(scalar)

from puxle.puzzles.rubikscube import RubiksCube

def get_action_map(puzzle):
    action_map = {}
    for a in range(puzzle.action_size):
        s = puzzle.action_to_string(a)
        action_map[s] = a
    return action_map

def parse_sequence(seq_str):
    # Split by space
    moves = seq_str.strip().split()
    expanded = []
    for m in moves:
        if m.endswith('2'):
            base = m[:-1]
            expanded.append(base)
            expanded.append(base)
        else:
            expanded.append(m)
    return expanded

def generate_state_from_sequence(puzzle, sequence, name=""):
    print(f"\n--- Generating {name} ---")
    print(f"Sequence: {sequence}")
    
    expanded_moves = parse_sequence(sequence)
    # No need to reverse/invert if we are generating FROM solved TO pattern.
    # The previous script was "reverse to find state that requires these moves to solve".
    # BUT, if the sequence is "how to generate pattern from solved", then applying it to solved gives the pattern.
    # The resulting state IS the pattern.
    # To solve it, you would apply the inverse.
    # So the state is correct.
    
    print(f"Expanded moves: {expanded_moves}")
    
    solve_config = puzzle.get_solve_config()
    current_state = solve_config.TargetState # Solved state
    
    action_map = get_action_map(puzzle)
    
    for m in expanded_moves:
        if m not in action_map:
            print(f"Error: Move {m} not found in action map")
            return None
        action = action_map[m]
        next_state, _ = puzzle.get_actions(solve_config, current_state, action)
        current_state = next_state
        
    faces = current_state.faces_unpacked
    faces_np = np.array(faces)
    flat = list(faces_np.flatten())
    print(f"Resulting State Faces (flattened):")
    print(flat)
    
    print("Visual check:")
    print(puzzle.get_string_parser()(current_state))
    return flat

def main():
    puzzle = RubiksCube(size=3, color_embedding=True)
    
    # 1. Superflip (20 HTM)
    # U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2
    superflip_seq = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
    generate_state_from_sequence(puzzle, superflip_seq, "Superflip")
    
    # 2. Fourspot (Using common sequence)
    # F2 R2 U2 F' B D2 L2 F B
    fourspot_seq = "F2 R2 U2 F' B D2 L2 F B"
    generate_state_from_sequence(puzzle, fourspot_seq, "Fourspot")

if __name__ == "__main__":
    main()
