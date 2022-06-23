import json
import subprocess
from string import digits

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar

from turbowsr import ROOT
AFLOW_EXECUTABLE = "aflow"

table = str.maketrans("", "", digits)


def get_proto_chemsys_params_from_struct(struct, aflow_executable=AFLOW_EXECUTABLE):
    """get prototype and parameters for pymatgen structure

    Args:
        struct (Structure): pymatgen Structure
        aflow_executable (PATH, optional): path to aflow executable. Defaults to AFLOW_EXECUTABLE.

    Returns:
        a tuple of the prototype, chemsys, params
    """
    poscar = Poscar(struct)

    cmd = f"{aflow_executable} --prototype --print=json cat"

    output = subprocess.run(
        cmd, input=poscar.get_string(), text=True, capture_output=True, shell=True
    )

    aflow_dict = json.loads(output.stdout)

    prototype = aflow_dict["aflow_prototype_label"]
    keys = aflow_dict["aflow_prototype_params_list"]
    values = aflow_dict["aflow_prototype_params_values"]
    chemsys = struct.composition.chemical_system

    params = dict(
        (k[0], v * values[0]) if "/a" in k else (k, v) for k, v in zip(keys, values)
    )

    # print(params)

    return prototype, chemsys, params


def get_struct_from_proto_chemsys_params(
    prototype, chemsys, params, aflow_executable=AFLOW_EXECUTABLE
):
    """get pymatgen structure given prototype and parameterss

    Args:
        prototype (str): structure prototype in aflow format
        chemsys (str): chemical system in pymatgen format
        params (dict): dictionary of prototype free-parameters
        aflow_executable (PATH, optional): path to aflow executable. Defaults to AFLOW_EXECUTABLE.

    Returns:
       pymatgen Structure reconstucted from prototype and params
    """
    # rescale lattice parameters for aflow input
    if "b" in params:
        params["b"] /= params["a"]
    if "c" in params:
        params["c"] /= params["a"]

    vals = ",".join(map(str, params.values()))
    chemsys = prototype.split("_")[0].translate(table) + ":" + chemsys.replace("-", ":")

    cmd = f"{aflow_executable} --proto={prototype}.{chemsys} --params={vals}"

    output = subprocess.run(cmd, text=True, capture_output=True, shell=True)

    if output.stderr:
        raise RuntimeError(output.stderr)

    return Structure.from_str(output.stdout, "poscar")


if __name__ == "__main__":
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from glob import glob

    matches = []
    for poscar in glob(f"{ROOT}/turbowsr/tests/POSCAR.*"):
        s = Structure.from_file(poscar)

        sm = StructureMatcher()
        matches.append(
            sm.fit(
                s,
                get_struct_from_proto_chemsys_params(
                    *get_proto_chemsys_params_from_struct(s)
                ),
            )
        )

    assert all(matches)
