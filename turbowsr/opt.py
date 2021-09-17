import numpy as np


from turbowsr.aflow import (
    get_proto_chemsys_params_from_struct,
    get_struct_from_proto_chemsys_params,
)


class TuRBOWSR:
    """[summary]

    Args:
        initial_struct ([type]): [description]
        model ([type]): [description]
    """

    LATTICE_BOUNDS = {"a": 50, "b/a": 10, "c/a": 10, "alpha": 360, "beta": 360, "gamma": 360}

    def __init__(self, initial_struct, model):
        self.initial_struct = initial_struct
        # self.optimised_struct = initial_struct

        aflow = get_proto_chemsys_params_from_struct(self.initial_struct)
        self.proto, self.chemsys, self.params = aflow

        self.model = model

        self.dim = len(self.params)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 1  # set the minimum for a lattice parameter to 1 \AA
        self.ub = np.array([
            TuRBOWSR.LATTICE_BOUNDS[k] if k in TuRBOWSR.LATTICE_BOUNDS.keys() else 1
            for k in self.params.keys()
        ])

    def __call__(self, x):
        """Evaluate the machine learning model for a set of structural parameters

        Args:
            x (ndarray): array of trial parameters

        Returns:
            model prediction to minimise
        """
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.params = dict(zip(self.params.keys(), x))

        print(self.params)

        try:
            s = get_struct_from_proto_chemsys_params(
                self.proto, self.chemsys, self.params
            )
            return self.model.predict_structure(s).item()
        except RuntimeError:
            return 66
