
def structural_factor_loss(atom_positions, sfc_instance):
    """Evaluates structure factor loss using only real-space global coordinates."""
    # Ensure positions are flattened/shaped correctly to (N, 3) for SFC_Jax
    X_local = atom_positions.reshape(-1, 3)

    # Delegate entirely to the differentiable structural engine
    return sfc_instance.compute_loss(X_local)
