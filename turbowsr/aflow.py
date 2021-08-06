# from megnet.utils.models import load_model

import json
import subprocess
from string import digits

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar


AFLOW_EXECUTABLE = "~/bin/aflow"

table = str.maketrans('', '', digits)


def get_proto_chemsys_params_from_struct(struct, aflow_executable=AFLOW_EXECUTABLE):
    poscar = Poscar(struct)

    cmd = f"{aflow_executable} --prototype --print=json cat"

    output = subprocess.run(
        cmd, input=poscar.get_string(), text=True, capture_output=True, shell=True
    )

    aflow_proto = json.loads(output.stdout)

    proto = aflow_proto["aflow_label"]
    chemsys = proto.split("_")[0].translate(table) + ":" + struct.composition.chemical_system.replace("-", ":")
    params = dict(zip(aflow_proto["aflow_parameter_list"], aflow_proto["aflow_parameter_values"]))

    return proto, chemsys, params



def get_struct_from_proto_chemsys_params(proto, chemsys, params, aflow_executable=AFLOW_EXECUTABLE):
    vals = ",".join(map(str, params.values()))
    cmd = f"{aflow_executable} --proto={proto}.{chemsys} --params={vals}"
    print(cmd)


if __name__ == "__main__":
        s = Structure.from_file("/home/reag2/PhD/turbowsr/POSCAR.mp-862972_AcAgAu2")

        print(get_proto_chemsys_params_from_struct(s))
        get_struct_from_proto_chemsys_params(*get_proto_chemsys_params_from_struct(s))