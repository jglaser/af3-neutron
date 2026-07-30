#!/usr/bin/env python3
import argparse
import copy
import functools
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List

import numpy as np

try:
    from Bio.Data.PDBData import protein_letters_3to1
    from Bio.PDB import MMCIFParser, PDBParser, Select
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio.PDB.Model import Model
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.Structure import Structure
except ImportError:
    print("Error: Biopython is required to run this script.", file=sys.stderr)
    print("Please install it using: pip install biopython", file=sys.stderr)
    sys.exit(1)

# Crystallisation agents and buffer salts: present in the deposited crystal
# because of how it was grown, not because they are part of the biological
# assembly. Passing them to AF3 as ligands asks it to predict binders that the
# protein has no reason to bind, and they occupy sites a real ligand may want.
# Override per-run with --keep_agents, or by naming one in --ligand_smiles.
DEFAULT_EXCLUDED_AGENTS = {
    "SO4", "PO4", "GOL", "EDO", "ACT", "CL", "NA", "K", "NH4", "CIT",
    "TRS", "MES", "HEPES", "PEG", "PGE", "PG4", "PE4", "1PE", "DTT", "DMS", "FMT"
}

# AF3 filters templates by revision date against its own --max_template_date.
# Deliberately a fixed past date, not today's: a template stamped in the future
# relative to that cutoff is silently dropped, and the symmetry templates this
# script writes must always survive the filter.
TEMPLATE_REVISION_DATE = "2026-07-06"


class ChainSelect(Select):
    """Filter class to isolate a single chain during MMCIFIO writing."""
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id

def get_assembly_operators(mmcif_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts Cartesian assembly/symmetry operators from the mmCIF dictionary."""
    if not mmcif_dict or "_pdbx_struct_oper_list.id" not in mmcif_dict:
        return [{"id": "1", "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "vector": [0.0, 0.0, 0.0]}]

    ids = mmcif_dict["_pdbx_struct_oper_list.id"]
    if isinstance(ids, str):
        ids = [ids]

    operators = []
    for i in range(len(ids)):
        mat = [[0.0]*3 for _ in range(3)]
        vec = [0.0]*3
        try:
            for r in range(1, 4):
                for c in range(1, 4):
                    key = f"_pdbx_struct_oper_list.matrix[{r}][{c}]"
                    val = mmcif_dict[key]
                    mat[r-1][c-1] = float(val[i] if isinstance(val, list) else val)
                v_key = f"_pdbx_struct_oper_list.vector[{r}]"
                v_val = mmcif_dict[v_key]
                vec[r-1] = float(v_val[i] if isinstance(v_val, list) else v_val)
            operators.append({"id": ids[i], "matrix": mat, "vector": vec})
        except (KeyError, ValueError, IndexError):
            continue

    if not operators:
        return [{"id": "1", "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "vector": [0.0, 0.0, 0.0]}]
    return operators

WATER_CODES = ("HOH", "DOD", "WAT")
PLACEHOLDER_SMILES = "PLACEHOLDER_SMILES"

# Persisted `ligand_id -> SMILES` table. The CCD is 49,835 components and AF3
# already ships it, so the network is a fallback rather than the default path.
# Overridable for tests and for running without a writable home directory.
CCD_CACHE_PATH = pathlib.Path(
    os.environ.get("AF3_NEUTRON_CCD_CACHE")
    or pathlib.Path.home() / ".cache" / "af3_neutron" / "ccd_smiles.json"
)


@functools.lru_cache(maxsize=1)
def _ccd_cache() -> Dict[str, str]:
    """Load the on-disk SMILES table. Corrupt or unreadable cache starts empty."""
    try:
        with open(CCD_CACHE_PATH, encoding="utf-8") as f:
            table = json.load(f)
        return table if isinstance(table, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_ccd_cache(table: Dict[str, str]) -> None:
    """Write the table, atomically, so a killed run cannot truncate it."""
    try:
        CCD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CCD_CACHE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(table, f, sort_keys=True)
        os.replace(tmp, CCD_CACHE_PATH)
    except OSError as e:
        print(f"Warning: could not write CCD cache: {e}", file=sys.stderr)


@functools.lru_cache(maxsize=1)
def _local_ccd():
    """AF3's bundled CCD, or None if alphafold3 is not importable.

    Imported lazily and softly: this script otherwise needs only Biopython, and
    constructing the CCD costs ~5 s, so it is worth avoiding on a cache hit.
    """
    try:
        from alphafold3.constants import chemical_components

        return chemical_components, chemical_components.Ccd()
    except Exception as e:
        print(f"Note: AF3 CCD unavailable ({e}); falling back to RCSB.", file=sys.stderr)
        return None


def _smiles_from_local_ccd(ligand_id: str) -> str | None:
    """SMILES for one component from AF3's bundled CCD."""
    loaded = _local_ccd()
    if loaded is None:
        return None
    chemical_components, ccd = loaded
    info = chemical_components.component_name_to_info(ccd=ccd, res_name=ligand_id)
    return getattr(info, "pdbx_smiles", None) if info else None


def fetch_pdb_ligand_smiles(ligand_id: str) -> str:
    """SMILES for a ligand, from the on-disk cache, then AF3's CCD, then RCSB."""
    ligand_id = ligand_id.upper().strip()
    if ligand_id in WATER_CODES:
        return "O"

    cache = _ccd_cache()
    if ligand_id in cache:
        return cache[ligand_id]

    smiles = _smiles_from_local_ccd(ligand_id)
    if smiles:
        cache[ligand_id] = smiles
        _save_ccd_cache(cache)
        print(f"Resolved '{ligand_id}' from AF3's local CCD.", file=sys.stderr)
        return smiles

    smiles = _fetch_ligand_smiles_from_rcsb(ligand_id)
    if smiles and smiles != PLACEHOLDER_SMILES:
        cache[ligand_id] = smiles
        _save_ccd_cache(cache)
    return smiles


def _fetch_ligand_smiles_from_rcsb(ligand_id: str) -> str:
    """Query the RCSB Chemical Component API. Needs network connectivity."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"
    print(f"Fetching SMILES for ligand '{ligand_id}' from PDB Component DB...", file=sys.stderr)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            descriptors = data.get("rcsb_chem_comp_descriptor", {})

            smiles = (descriptors.get("SMILES") or
                          descriptors.get("SMILES_stereo") or
                          descriptors.get("smiles") or
                          descriptors.get("smiles_stereo"))

            return smiles
    except urllib.error.HTTPError as e:
        print(f"Warning: Ligand '{ligand_id}' not found in PDB database (HTTP Error {e.code}).", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Network error fetching ligand '{ligand_id}': {str(e)}", file=sys.stderr)

    return PLACEHOLDER_SMILES

def generate_af3_json(input_path: str, job_name: str, output_dir: str, remove_water: bool, keep_agents: bool, cmd_smiles: Dict[str, str], model_seed: int = 1) -> Dict[str, Any]:
    """Generates the AF3 JSON input and applies symmetry operators using MMCIF2Dict."""
    file_ext = os.path.splitext(input_path)[1].lower()
    mmcif_dict = None
    structure_id = os.path.splitext(os.path.basename(input_path))[0]

    if file_ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(structure_id, input_path)
        try:
            mmcif_dict = MMCIF2Dict(input_path)
        except Exception as e:
            print(f"Warning: Failed to parse mmCIF dictionary layout: {str(e)}", file=sys.stderr)
    elif file_ext in [".pdb"]:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(structure_id, input_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    model = structure[0]
    operators = get_assembly_operators(mmcif_dict)

    sequences_json: List[Dict[str, Any]] = []
    fetched_smiles_cache: Dict[str, str] = {}

    used_ids = set()
    id_counter = 0

    def next_alpha_id() -> str:
        nonlocal id_counter
        while True:
            res = ""
            temp = id_counter
            while temp >= 0:
                res = chr(temp % 26 + 65) + res
                temp = temp // 26 - 1
            id_counter += 1
            if res not in used_ids:
                used_ids.add(res)
                return res

    # Pre-pass to reserve clean, existing protein chain IDs
    reserved_chain_ids = {}
    for chain in model:
        has_protein = any(residue.id[0] == " " and is_aa(residue) for residue in chain)
        if has_protein:
            c_id = chain.id.strip()
            if c_id.isalpha() and c_id.isupper():
                reserved_chain_ids[chain.id] = c_id
                used_ids.add(c_id)

    # Loop through each crystallographic symmetry/assembly transformation operator
    for op_idx, oper in enumerate(operators):
        is_first_op = (op_idx == 0)

        # 1. Process and Transform Protein Chains
        for chain in model:
            protein_seq = []
            for residue in chain:
                if residue.id[0] == " " and is_aa(residue):
                    res_name = residue.get_resname().strip().upper()
                    one_letter = protein_letters_3to1.get(res_name) or protein_letters_3to1.get(res_name.title(), "X")
                    protein_seq.append(one_letter)

            if protein_seq:
                sequence_str = "".join(protein_seq)
                seq_length = len(sequence_str)
                indices = list(range(seq_length))

                c_id = chain.id.strip()
                if is_first_op and c_id.isalpha() and c_id.isupper() and c_id not in used_ids:
                    assigned_id = c_id
                    used_ids.add(assigned_id)
                else:
                    assigned_id = next_alpha_id()

                template_filename = f"{structure_id}_template_chain_{assigned_id}.cif"
                template_filepath = os.path.join(output_dir, template_filename)

                # Deepcopy and apply Cartesian coordinate rotation matrices
                transformed_chain = copy.deepcopy(chain)
                transformed_chain.id = assigned_id

                # Built once per operator, not per atom: rebuilding these inside the
                # atom loop costs more than the hand-expanded multiply it replaced
                # (4.23 ms vs 2.30 ms per 2000 atoms). Hoisted, it is 2.17 ms.
                matrix = np.asarray(oper["matrix"], dtype=float)
                vector = np.asarray(oper["vector"], dtype=float)

                for residue in transformed_chain:
                    for atom in residue:
                        atom.set_coord(matrix @ atom.get_coord() + vector)

                # Save out to its isolated template destination
                dummy_struct = Structure("temp")
                dummy_model = Model(0)
                dummy_struct.add(dummy_model)
                dummy_model.add(transformed_chain)

                cif_io_temp = MMCIFIO()
                cif_io_temp.set_structure(dummy_struct)
                cif_io_temp.save(template_filepath, ChainSelect(assigned_id))

                # Write version compliance loops
                with open(template_filepath, "a") as f:
                    f.write("\n#\n")
                    f.write("loop_\n")
                    f.write("_pdbx_audit_revision_history.ordinal\n")
                    f.write("_pdbx_audit_revision_history.revision_date\n")
                    f.write(f"1 {TEMPLATE_REVISION_DATE}\n")
                    f.write("#\n")

                print(f"Exported symmetry template file: {template_filepath}", file=sys.stderr)

                chain_json = {
                    "protein": {
                        "id": [assigned_id],
                        "sequence": sequence_str,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [
                            {
                                "mmcifPath": template_filename,
                                "queryIndices": indices,
                                "templateIndices": indices
                            }
                        ]
                    }
                }
                sequences_json.append(chain_json)

        # 2. Process and Map Ligands per Symmetry Copy
        for chain in model:
            for residue in chain:
                het_flag, res_seq, icode = residue.id
                if het_flag.startswith("H_"):
                    res_name = residue.get_resname().strip().upper()

                    is_water = res_name in ["HOH", "DOD", "WAT"]
                    if is_water and remove_water:
                        continue

                    if res_name in DEFAULT_EXCLUDED_AGENTS and res_name not in cmd_smiles and not keep_agents:
                        continue

                    if res_name in cmd_smiles:
                        smiles = cmd_smiles[res_name]
                    elif is_water:
                        smiles = "O"
                    elif res_name in fetched_smiles_cache:
                        smiles = fetched_smiles_cache[res_name]
                    else:
                        smiles = fetch_pdb_ligand_smiles(res_name)
                        fetched_smiles_cache[res_name] = smiles

                    ligand_id = next_alpha_id()

                    ligand_json = {
                        "ligand": {
                            "id": [ligand_id],
                            "smiles": smiles
                        }
                    }
                    sequences_json.append(ligand_json)

    return {
        "name": job_name,
        "modelSeeds": [model_seed],
        "dialect": "alphafold3",
        "version": 2,
        "sequences": sequences_json
    }

def main():
    parser = argparse.ArgumentParser(description="Generate structure-factor guided AF3 v2 JSON inputs with full symmetry packing.")
    parser.add_argument("-i", "--input", required=True, help="Path to coordinate file (.pdb, .cif, .mmcif)")
    parser.add_argument("-o", "--output", required=True, help="Path to write the target output .json file")
    parser.add_argument("-n", "--name", default="af3_refinement_job", help="Job name prefix inside the JSON bundle")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed integer for the AF3 run")
    parser.add_argument("--remove-water", action="store_true", help="Filter out crystallographic water instances")
    parser.add_argument("--keep-agents", action="store_true", help="Do not filter out common crystallization agents/salts (SO4, GOL, etc.)")
    parser.add_argument("--ligand-smiles", help="Manual SMILES overrides as key=value pairs (e.g., BZB=SMILES,PO4=SMILES)")

    args = parser.parse_args()

    cmd_smiles = {}
    if args.ligand_smiles:
        for pair in args.ligand_smiles.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                cmd_smiles[k.strip().upper()] = v.strip()

    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."

    try:
        output_data = generate_af3_json(
            input_path=args.input,
            job_name=args.name,
            output_dir=output_dir,
            remove_water=args.remove_water,
            keep_agents=args.keep_agents,
            cmd_smiles=cmd_smiles,
            model_seed=args.seed
        )
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Successfully generated refinement JSON configuration at: {args.output}")
    except Exception as e:
        print(f"Generation Failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
