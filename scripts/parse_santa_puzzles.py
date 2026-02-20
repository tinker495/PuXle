import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def parse_state(state_str: str) -> np.ndarray:
    """Parse a semicolon-separated state string into a numpy array."""
    return np.array(state_str.split(';'))

def get_color_mapping(solution_state: np.ndarray) -> Dict[str, int]:
    """
    Create a mapping from state symbols (e.g., 'A', 'B') to integers (0..N).
    Assumes the solution state defines the canonical order of colors.
    """
    unique_colors = sorted(np.unique(solution_state))
    return {color: i for i, color in enumerate(unique_colors)}

def map_state(state: np.ndarray, mapping: Dict[str, int]) -> np.ndarray:
    """Map state symbols to integers using the provided mapping."""
    return np.vectorize(mapping.get)(state).astype(np.uint8)

def parse_santa_dataset(
    puzzles_path: str | Path,
    puzzle_info_path: str | Path | None = None,
    filter_type: str | None = "cube"
) -> Dict[str, Any]:
    """
    Parse the Santa 2023 puzzles.csv file.
    
    Args:
        puzzles_path: Path to puzzles.csv
        puzzle_info_path: Path to puzzle_info.csv (optional)
        filter_type: Prefix to filter puzzle types (e.g., 'cube'), or None for all.
        
    Returns:
        A dictionary containing parsed puzzles grouped by type.
    """
    df = pd.read_csv(puzzles_path)
    
    if filter_type:
        df = df[df['puzzle_type'].str.startswith(filter_type)]
        
    parsed_data = {}
    
    for puzzle_type, group in df.groupby('puzzle_type'):
        # Parse size from string (e.g., "cube_3/3/3" -> 3)
        if puzzle_type.startswith("cube"):
            try:
                # Extract dimensions
                dims = puzzle_type.split('_')[1].split('/')
                size = int(dims[0])
            except (IndexError, ValueError):
                size = None
        else:
            size = None
            
        samples = []
        for _, row in group.iterrows():
            initial_raw = parse_state(row['initial_state'])
            solution_raw = parse_state(row['solution_state'])
            
            # Create color mapping based on solution state
            # Note: different puzzles of same type might use different symbols?
            # We map per puzzle to be safe, or could check global consistency.
            mapping = get_color_mapping(solution_raw)
            
            initial_mapped = map_state(initial_raw, mapping)
            solution_mapped = map_state(solution_raw, mapping)
            
            samples.append({
                'id': row['id'],
                'initial_state': initial_mapped,
                'solution_state': solution_mapped,
                'wildcards': row['num_wildcards'],
                'color_mapping': mapping
            })
            
        parsed_data[puzzle_type] = {
            'size': size,
            'samples': samples
        }
        
    return parsed_data

if __name__ == "__main__":
    # Example usage
    base_path = Path(__file__).resolve().parents[1] / "puxle" / "data" / "santa-2023"
    puzzles_file = base_path / "puzzles.csv"
    
    if puzzles_file.exists():
        print(f"Parsing {puzzles_file}...")
        data = parse_santa_dataset(puzzles_file, filter_type="cube")
        for p_type, content in data.items():
            print(f"Type: {p_type}, Size: {content['size']}, Count: {len(content['samples'])}")
            # Print first sample check
            sample = content['samples'][0]
            print(f"  Sample 0 ID: {sample['id']}")
            print(f"  Initial shape: {sample['initial_state'].shape}")
    else:
        print(f"File not found: {puzzles_file}")

