"""Placement of the model into the data's crystal frame.

Three things have to be right before any structure factor is meaningful, and
each one fails silently on its own:

1. the reference must be expressed in the DATA's unit cell (fractional transfer),
2. the CA correspondence must be by sequence, not by array index or residue id,
3. the model must be in the same SETTING as the data -- the alternative settings
   of a space group are indistinguishable from the symmetry description alone.

The first two are exercised on synthetic structures so they always run.  The
third is checked both as pure group theory (always runs) and against the real
4BD0/4BD1 pair when those files are present.
"""

import os

import biotite.structure as struc
import gemmi
import numpy as np
import pytest

from af3_neutron.sfc_adapter import (
    _box_from_cell,
    _match_ca_by_sequence,
    reindexing_operators,
)

CELL_4BD1 = (73.429, 73.429, 99.112, 90.0, 90.0, 120.0)
CELL_4BD0 = (72.500, 72.500, 97.670, 90.0, 90.0, 120.0)

FM = os.path.join(os.path.dirname(__file__), "..", "..", "forward_model")
HAVE_DATA = all(
    os.path.exists(os.path.join(FM, f)) for f in ("4BD0.cif", "4BD1.cif", "4BD1.mtz")
)
requires_data = pytest.mark.skipif(
    not HAVE_DATA, reason=f"4BD0/4BD1 reference data not found under {FM}"
)


# --------------------------------------------------------------------------
# 1. cell handling
# --------------------------------------------------------------------------

def test_box_from_cell_matches_biotite_in_radians():
    """`_box_from_cell` takes DEGREES; biotite's own helper takes RADIANS.

    This is the trap `_box_from_cell` exists to close.  Feeding degrees to
    `vectors_from_unitcell` does not raise -- it returns a plausible-looking
    triclinic box -- so the failure is silent and every transferred coordinate
    is wrong.
    """
    got = _box_from_cell(CELL_4BD1)
    want = np.asarray(
        struc.vectors_from_unitcell(*CELL_4BD1[:3], *np.deg2rad(CELL_4BD1[3:]))
    )
    assert np.allclose(got, want, atol=1e-4)

    # and the degrees-fed version really is different, i.e. the trap is real
    wrong = np.asarray(struc.vectors_from_unitcell(*CELL_4BD1))
    assert not np.allclose(wrong, want, atol=1e-2)


def test_box_rows_are_cell_vectors():
    """biotite's convention is cart = frac @ box, i.e. cell vectors as ROWS."""
    box = _box_from_cell(CELL_4BD1)
    lengths = np.linalg.norm(box, axis=1)
    assert np.allclose(lengths, CELL_4BD1[:3], atol=1e-3)
    cos_gamma = np.dot(box[0], box[1]) / (lengths[0] * lengths[1])
    assert np.isclose(np.rad2deg(np.arccos(cos_gamma)), CELL_4BD1[5], atol=1e-3)


def test_fractional_transfer_preserves_fractional_coordinates():
    """The whole point: fractional coordinates are invariant under the transfer."""
    rng = np.random.default_rng(0)
    frac = rng.uniform(size=(500, 3))
    box_a, box_b = _box_from_cell(CELL_4BD0), _box_from_cell(CELL_4BD1)

    cart_a = frac @ box_a
    cart_b = (cart_a @ np.linalg.inv(box_a)) @ box_b

    assert np.allclose(cart_b @ np.linalg.inv(box_b), frac, atol=1e-12)
    # and it is not a no-op: the 1.3-1.5% cell difference really does move atoms
    assert np.linalg.norm(cart_b - cart_a, axis=1).mean() > 0.3


# --------------------------------------------------------------------------
# 2. CA correspondence
# --------------------------------------------------------------------------

def _ca_array(res_names, res_ids, chain="A", coords=None):
    a = struc.AtomArray(len(res_names))
    a.coord = (np.zeros((len(res_names), 3), dtype=np.float32)
               if coords is None else np.asarray(coords, dtype=np.float32))
    a.atom_name = np.array(["CA"] * len(res_names))
    a.element = np.array(["C"] * len(res_names))
    a.res_name = np.array(res_names)
    a.res_id = np.array(res_ids)
    a.chain_id = np.array([chain] * len(res_names))
    return a


SEQ3 = ["ALA", "GLY", "SER", "THR", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR",
        "TRP", "HIS", "LYS", "ARG", "ASP", "GLU", "ASN", "GLN", "MET", "CYS"]


def test_matching_survives_different_residue_numbering():
    """The 4BD0/4BD1 case: identical sequence, different numbering scheme.

    The reference uses Ambler numbering (27..), the oracle numbers its own input
    from 1.  Keying on res_id pairs residue 27 with residue 27 -- a 26-residue
    register slip that still yields plenty of "matches", so a count-based check
    passes and the error only surfaces as a large RMSD.
    """
    ref = _ca_array(SEQ3, list(range(27, 27 + len(SEQ3))))
    orc = _ca_array(SEQ3, list(range(1, 1 + len(SEQ3))))

    idx_ref, idx_orc, how = _match_ca_by_sequence(ref, orc)

    assert len(idx_ref) == len(SEQ3)
    assert np.array_equal(idx_ref, idx_orc), "must be the identity pairing"
    assert "sequence alignment" in how


def test_matching_opens_gaps_for_missing_residues():
    """A reference missing a disordered loop must not shift the register."""
    keep = [i for i in range(len(SEQ3)) if i not in (5, 6, 7)]
    ref = _ca_array([SEQ3[i] for i in keep], [27 + i for i in keep])
    orc = _ca_array(SEQ3, list(range(1, 1 + len(SEQ3))))

    idx_ref, idx_orc, _ = _match_ca_by_sequence(ref, orc)

    assert len(idx_ref) == len(keep)
    # every matched pair must name the same residue -- that is what "register" means
    assert list(ref.res_name[idx_ref]) == list(orc.res_name[idx_orc])
    assert np.array_equal(idx_orc, np.array(keep))


def test_matching_assigns_chains_by_similarity_not_by_file_order():
    """Two chains listed in opposite orders must still pair correctly."""
    seq_x, seq_y = SEQ3, SEQ3[::-1]
    ref = _ca_array(seq_x + seq_y,
                    list(range(1, 1 + len(seq_x))) + list(range(1, 1 + len(seq_y))))
    ref.chain_id = np.array(["P"] * len(seq_x) + ["Q"] * len(seq_y))
    orc = _ca_array(seq_y + seq_x,
                    list(range(1, 1 + len(seq_y))) + list(range(1, 1 + len(seq_x))))
    orc.chain_id = np.array(["A"] * len(seq_y) + ["B"] * len(seq_x))

    idx_ref, idx_orc, _ = _match_ca_by_sequence(ref, orc)

    assert len(idx_ref) == len(seq_x) + len(seq_y)
    assert list(ref.res_name[idx_ref]) == list(orc.res_name[idx_orc])


# --------------------------------------------------------------------------
# 3. alternative settings (reindexing operators)
# --------------------------------------------------------------------------

def test_reindexing_operators_for_p3221():
    """6/mmm has order 24, P 32 2 1 has 6 rotations -> 4 cosets -> 2 after Friedel."""
    ops = reindexing_operators(CELL_4BD1, "P 32 2 1")

    assert len(ops) == 2
    assert ops[0][1] == "x,y,z", "identity must come first"
    assert ops[1][1] == "-x,-y,z", f"expected the 2-fold along c, got {ops[1][1]}"


def test_reindexing_operators_are_lattice_symmetries():
    """Every returned W must preserve the metric tensor exactly."""
    from af3_neutron.sfc_adapter import _metric_tensor

    for cell, sg in ((CELL_4BD1, "P 32 2 1"), ((60.0, 70.0, 80.0, 90, 90, 90), "P 21 21 21")):
        G = _metric_tensor(cell)
        for W, label in reindexing_operators(cell, sg):
            W = np.asarray(W, dtype=float)
            assert np.allclose(W.T @ G @ W, G, atol=1e-6 * np.abs(G).max()), label
            assert abs(abs(np.linalg.det(W)) - 1.0) < 1e-9, label


def test_generic_orthorhombic_has_no_alternative_setting():
    """P 21 21 21 on a generic cell: mmm (order 8) / 222 (order 4) = 2 cosets,
    and Friedel identifies them because mmm = 222 x {I, -I}.  So there is nothing
    to choose and the resolver must not invent a candidate."""
    ops = reindexing_operators((60.0, 70.0, 80.0, 90.0, 90.0, 90.0), "P 21 21 21")
    assert [label for _, label in ops] == ["x,y,z"]


def test_metrically_special_cell_gains_a_setting():
    """The same space group on an a == b cell has a tetragonal LATTICE, so an
    extra operator appears -- exactly the pseudo-symmetry that makes indexing
    ambiguous in practice.  The enumeration must be driven by the cell metric,
    not by the space group alone."""
    ops = reindexing_operators((60.0, 60.0, 80.0, 90.0, 90.0, 90.0), "P 21 21 21")
    assert [label for _, label in ops] == ["x,y,z", "-y,-x,-z"]


def test_monoclinic_has_no_alternative_setting():
    ops = reindexing_operators((60.0, 70.0, 80.0, 90.0, 100.0, 90.0), "P 1 21 1")
    assert [label for _, label in ops] == ["x,y,z"]


# --------------------------------------------------------------------------
# 4. against the real depositions
# --------------------------------------------------------------------------

@requires_data
def test_4bd0_and_4bd1_differ_by_a_reindexing_operator():
    """The finding this machinery exists for.

    4BD0 (X-ray) and 4BD1 (neutron) are the same protein in the same space group
    and near-identical cells, but their ASUs are related by -x,-y,z with zero
    translation.  That operator is in the NORMALIZER of P 32 2 1, not in the
    group, so nothing about the symmetry description reveals it -- and it changes
    |F|, so a model placed on one cannot fit the other's data.
    """
    import biotite.structure.io.pdbx as pdbx

    def ca(path, to_cell=None):
        a = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
        if to_cell is not None:
            box = np.asarray(a.box, dtype=np.float64)
            a.coord = (a.coord @ np.linalg.inv(box) @ _box_from_cell(to_cell))
        m = (a.atom_name == "CA") & struc.filter_amino_acids(a)
        return np.asarray(a.coord[m], dtype=np.float64)

    x0 = ca(os.path.join(FM, "4BD0.cif"), to_cell=CELL_4BD1)
    x1 = ca(os.path.join(FM, "4BD1.cif"))
    assert len(x0) == len(x1) == 261

    # as deposited they are far apart ...
    assert np.sqrt(((x0 - x1) ** 2).sum(1).mean()) > 50.0

    # ... but applying -x,-y,z to 4BD0 lands it on 4BD1, with NO superposition
    uc = gemmi.UnitCell(*CELL_4BD1)
    F = np.array(uc.frac.mat.tolist())
    orth = np.array(uc.orth.mat.tolist())
    M = np.diag([-1.0, -1.0, 1.0])
    x0M = ((F @ x0.T).T @ M.T) @ orth.T
    assert np.sqrt(((x0M - x1) ** 2).sum(1).mean()) < 1.0
