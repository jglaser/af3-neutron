import dataclasses
import jax
import jax.numpy as jnp
from typing import Any

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HostEmbeddings:
    """Carries frozen representations extracted from the native AlphaFold 3 trunk layer."""
    pair: jnp.ndarray
    single: jnp.ndarray
    target_feat: jnp.ndarray

@dataclasses.dataclass(frozen=True)
class OracleMapping:
    """Defines the structural translation coordinates mapping AlphaFold 3 to the physical oracle."""
    num_atoms: int
    heavy_indices: jnp.ndarray
    source_indices: jnp.ndarray
    initial_coordinates: jnp.ndarray

@dataclasses.dataclass
class Oracle:
    """Carries the unified chemical topology and coordinate layout profiles."""
    mapping: OracleMapping
    atoms: Any  # biotite.structure.AtomArray

@dataclasses.dataclass(frozen=True)
class Conformations:
    """Carries the final refined Cartesian coordinates from the diffusion trajectory."""
    atom_positions: jnp.ndarray

    def __getitem__(self, index):
        """Allows direct subscripting (e.g., conformations[0]) to extract specific sample slices."""
        return self.atom_positions[index]
