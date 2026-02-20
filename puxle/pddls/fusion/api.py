from typing import List, Optional, Tuple
from pathlib import Path
import os
import pddl
from pddl.core import Domain

from puxle import PDDL
from puxle.pddls.fusion.domain_fusion import DomainFusion
from puxle.pddls.fusion.action_modifier import ActionModifier, FusionParams
from puxle.pddls.fusion.problem_generator import ProblemGenerator

def _domain_type_parent_map(domain: Domain) -> dict[str, str]:
    """Return {type_name: parent_type_name} for compatibility checks."""
    mapping: dict[str, str] = {}
    domain_types = getattr(domain, "types", None)
    if isinstance(domain_types, dict):
        for t, parent in domain_types.items():
            t_name = str(t)
            parent_name = "object" if parent is None else str(parent)
            mapping[t_name] = parent_name
    elif domain_types:
        for t in domain_types:
            mapping[str(t)] = "object"
    return mapping

def fuse_and_load(
    domain_paths: List[str], 
    params: Optional[FusionParams] = None,
    name: str = "fused-domain",
    **kwargs
) -> PDDL:
    """
    Fuses multiple PDDL domains and returns a PuXle PDDL environment.
    
    Args:
        domain_paths: List of paths to PDDL domain files.
        params: Fusion and modification parameters.
        name: Name of the resulting domain.
        **kwargs: Additional arguments for PDDL constructor.

    Returns:
        A PDDL instance with the fused domain and a simple/empty problem.
    """
    if params is None:
        params = FusionParams()

    # 1. Parse all domains
    domains = [pddl.parse_domain(d) for d in domain_paths]

    # 2. Fuse Domains (Structural Merge)
    fusion_engine = DomainFusion()
    fused_domain = fusion_engine.fuse_domains(domains, name=name)

    # 3. Modify Actions (Stochastic Dynamics)
    # We need all predicates for sampling new conditions/effects
    all_predicates = list(fused_domain.predicates)
    types_map = _domain_type_parent_map(fused_domain)
    
    modifier = ActionModifier(params)
    modified_actions = modifier.modify_actions(
        list(fused_domain.actions), 
        all_predicates, 
        types_map
    )

    # Replace actions in domain
    # pddl.Domain is immutable-ish but we can construct new one
    final_domain = Domain(
        name=fused_domain.name,
        requirements=fused_domain.requirements,
        types=fused_domain.types,
        constants=fused_domain.constants,
        predicates=fused_domain.predicates,
        actions=modified_actions
    )

    # 4. Generate a default Problem
    # We need a problem to initialize the PDDL env (PuXle requirement)
    generator = ProblemGenerator(seed=params.seed)
    # Just generate a small sample problem logic
    problem = generator.generate_problem(final_domain, num_objects=5, walk_length=5)

    # 5. Initialize PDDL Environment
    # We pass objects directly instead of paths
    return PDDL(domain=final_domain, problem=problem, **kwargs)

def generate_benchmark(
    domain_paths: List[str],
    output_dir: str,
    count: int = 10,
    params: Optional[FusionParams] = None,
    difficulty_depth_range: Tuple[int, int] = (5, 20)
) -> None:
    """
    Generates a suite of PDDL domain/problem files for external benchmarks.
    """
    if params is None:
        params = FusionParams()

    os.makedirs(output_dir, exist_ok=True)
    
    # Fusion (Single domain variant for the batch?)
    # Or do we want unique domain variant per problem?
    # Usually benchmarks have fixed domain, multiple problems.
    # So we generate ONE fused domain.
    
    domains = [pddl.parse_domain(d) for d in domain_paths]
    fusion_engine = DomainFusion()
    fused_domain = fusion_engine.fuse_domains(domains, name="fused-benchmark")
    
    all_predicates = list(fused_domain.predicates)
    types_map = _domain_type_parent_map(fused_domain)
    modifier = ActionModifier(params)
    modified_actions = modifier.modify_actions(list(fused_domain.actions), all_predicates, types_map)
    
    final_domain = Domain(
        name=fused_domain.name,
        requirements=fused_domain.requirements,
        types=fused_domain.types,
        constants=fused_domain.constants,
        predicates=fused_domain.predicates,
        actions=modified_actions
    )
    
    # Write Domain
    domain_file = os.path.join(output_dir, "domain.pddl")
    with open(domain_file, "w") as f:
         f.write(pddl.formatter.domain_to_string(final_domain))
         
    # Generate Problems
    rng = ProblemGenerator(seed=params.seed)
    
    start_depth, end_depth = difficulty_depth_range
    step = (end_depth - start_depth) / count if count > 1 else 0
    
    for i in range(count):
        depth = int(start_depth + i * step)
        prob_name = f"prob-{i:02d}"
        problem = rng.generate_problem(final_domain, num_objects=5 + (i//3), walk_length=depth, problem_name=prob_name)
        
        prob_file = os.path.join(output_dir, f"{prob_name}.pddl")
        with open(prob_file, "w") as f:
            f.write(pddl.formatter.problem_to_string(problem))


    print(f"Generated benchmark in {output_dir}: 1 domain, {count} problems.")

def iterative_fusion(
    base_domains: List[str],
    depth: int,
    params: FusionParams,
    name_prefix: str = "fused"
) -> Domain:
    """
    Performs iterative fusion to increase complexity.
    
    Args:
        base_domains: Initial domain paths
        depth: Number of fusion iterations (1 = basic fusion)
        params: Fusion and modification parameters
        name_prefix: Prefix for generated domain names
    
    Returns:
        Final fused domain after 'depth' iterations
    """
    if depth < 1:
        raise ValueError("Depth must be >= 1")
    
    # First iteration: fuse base domains
    domains = [pddl.parse_domain(d) for d in base_domains]
    fusion_engine = DomainFusion()
    current_domain = fusion_engine.fuse_domains(domains, name=f"{name_prefix}-d1")
    
    # Apply modifications
    all_predicates = list(current_domain.predicates)
    types_map = _domain_type_parent_map(current_domain)
    modifier = ActionModifier(params)
    modified_actions = modifier.modify_actions(
        list(current_domain.actions), all_predicates, types_map
    )
    
    current_domain = Domain(
        name=current_domain.name,
        requirements=current_domain.requirements,
        types=current_domain.types,
        constants=current_domain.constants,
        predicates=current_domain.predicates,
        actions=modified_actions
    )
    
    # Subsequent iterations: fuse current with itself to increase density
    # This is a simple interpretation of iterative fusion: 
    # merging the domain with itself increases action count (if renaming works) 
    # or just allows re-application of stochastic modifiers on a "denser" base?
    # PDDLFuse paper suggests fusing generated domains.
    for i in range(2, depth + 1):
        # We fuse the current domain with itself. 
        # Since we implemented renaming, this doubles the actions (original + renamed copy).
        # This rapidly increases domain size.
        current_domain = fusion_engine.fuse_domains(
            [current_domain, current_domain], 
            name=f"{name_prefix}-d{i}"
        )
        
        # Re-apply modifications to the larger set
        all_predicates = list(current_domain.predicates)
        types_map = _domain_type_parent_map(current_domain)
        modified_actions = modifier.modify_actions(
            list(current_domain.actions), all_predicates, types_map
        )
        
        current_domain = Domain(
            name=current_domain.name,
            requirements=current_domain.requirements,
            types=current_domain.types,
            constants=current_domain.constants,
            predicates=current_domain.predicates,
            actions=modified_actions
        )
    
    return current_domain

def generate_benchmark_with_varying_depth(
    base_domains: List[str],
    output_dir: str,
    depth_range: Tuple[int, int] = (1, 3),
    problems_per_depth: int = 10,
    params: Optional[FusionParams] = None
) -> None:
    """
    Generates benchmarks across a range of fusion depths.
    
    Args:
        base_domains: List of PDDL domain file paths
        output_dir: Root directory for output
        depth_range: (min_depth, max_depth) inclusive
        problems_per_depth: Number of problems per depth
        params: Fusion parameters
    """
    if params is None:
        params = FusionParams()
        
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    min_d, max_d = depth_range
    for d in range(min_d, max_d + 1):
        # Create sub-directory for this depth
        depth_dir = root / f"depth_{d}"
        depth_dir.mkdir(exist_ok=True)
        
        # 1. Create fused domain for this depth
        # We use iterative fusion
        # Note: iterative_fusion parses domains inside.
        print(f"Generating depth {d} domain...")
        fused_domain = iterative_fusion(
            base_domains, 
            depth=d, 
            params=params, 
            name_prefix=f"fused_d{d}"
        )
        
        # Save domain
        domain_path = depth_dir / "domain.pddl"
        with open(domain_path, "w") as f:
            f.write(pddl.formatter.domain_to_string(fused_domain))
            
        print(f"  Saved domain to {domain_path}")
        
        # 2. Generate problems
        generator = ProblemGenerator(seed=params.seed + d) # vary seed by depth
        
        for i in range(problems_per_depth):
            # We vary complexity slightly by varying problem size?
            # Or keep constant size but harder domain structure.
            # Let's keep size somewhat constant or scaled by depth?
            # Let's keep constant for comparable size benchmarks.
            num_objs = 5 + d * 2 # Scale objects with depth?
            walk_len = 10 + d * 5
            
            p_name = f"prob_d{d}_{i:02d}"
            problem = generator.generate_problem(
                fused_domain, 
                num_objects=num_objs, 
                walk_length=walk_len, 
                problem_name=p_name
            )
            
            p_path = depth_dir / f"problem_{i:02d}.pddl"
            with open(p_path, "w") as f:
                f.write(pddl.formatter.problem_to_string(problem))
                
        print(f"  Generated {problems_per_depth} problems in {depth_dir}")
