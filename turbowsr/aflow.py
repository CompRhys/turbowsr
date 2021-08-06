import json
import subprocess
from string import digits

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar


AFLOW_EXECUTABLE = "~/src/AFLOW/aflow"

table = str.maketrans('', '', digits)


def get_proto_chemsys_params_from_struct(struct, aflow_executable=AFLOW_EXECUTABLE):
    """[summary]

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
    chemsys = struct.composition.chemical_system
    params = dict(zip(aflow_dict["aflow_prototype_params_list"], aflow_dict["aflow_prototype_params_values"]))

    return prototype, chemsys, params


def get_struct_from_proto_chemsys_params(prototype, chemsys, params, aflow_executable=AFLOW_EXECUTABLE):
    """[summary]

    Args:
        prototype (str): structure prototype in aflow format
        chemsys (str): chemical system in pymatgen format
        params (dict): dictionary of prototype free-parameters
        aflow_executable (PATH, optional): path to aflow executable. Defaults to AFLOW_EXECUTABLE.

    Returns:
       pymatgen Structure reconstucted from prototype and params
    """
    vals = ",".join(map(str, params.values()))
    chemsys = prototype.split("_")[0].translate(table) + ":" + chemsys.replace("-", ":")

    cmd = f"{aflow_executable} --proto={prototype}.{chemsys} --params={vals}"

    output = subprocess.run(cmd, text=True, capture_output=True, shell=True)

    return Structure.from_str(output.stdout, "poscar")


if __name__ == "__main__":
        from pymatgen.analysis.structure_matcher import StructureMatcher

        s = Structure.from_file("/home/reag2/PhD/turbowsr/POSCAR.mp-554710_AgAsC4S8(N2F3)2")

        sm = StructureMatcher()
        print(sm.fit(s, get_struct_from_proto_chemsys_params(*get_proto_chemsys_params_from_struct(s))))
