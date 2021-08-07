import numpy as np

from turbo import Turbo1

from turbowsr.aflow import (
    get_proto_chemsys_params_from_struct,
    get_struct_from_proto_chemsys_params,
)


class TuRBOWSR:

    LATTICE_BOUNDS = {"a": 50, "b/a": 10, "c/a": 10, "alpha": 360, "beta": 360, "gamma": 360}

    def __init__(self, initial_struct, model):
        self.initial_struct = initial_struct
        # self.optimised_struct = initial_struct

        aflow = get_proto_chemsys_params_from_struct(self.initial_struct)
        self.proto, self.chemsys, self.params = aflow

        self.model = model

        self.dim = len(self.params)
        self.lb = np.zeros(self.dim)
        self.ub = np.array([TuRBOWSR.LATTICE_BOUNDS[k] if k in TuRBOWSR.LATTICE_BOUNDS.keys() else 1 for k in self.params.keys()])

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.params = dict(zip(self.params.keys(), x))

        s = get_struct_from_proto_chemsys_params(self.proto, self.chemsys, self.params)

        try:
            return self.model.predict_structure(s)
        except RuntimeError:
            return 999


if __name__ == "__main__":
    from pymatgen.core.structure import Structure
    import matplotlib
    import matplotlib.pyplot as plt

    from megnet.utils.models import load_model

    s = Structure.from_file("/home/reag2/PhD/turbowsr/turbowsr/tests/POSCAR.mp-554710_AgAsC4S8(N2F3)2")

    model = load_model("Bandgap_MP_2018")

    f = TuRBOWSR(s, model)

    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=20,  # Number of initial bounds from an Latin hypercube design
        max_evals=1000,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )

    turbo1.optimize()

    X = turbo1.X  # Evaluated points
    fX = turbo1.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]

    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

    fig = plt.figure(figsize=(7, 5))
    matplotlib.rcParams.update({'font.size': 16})
    plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
    plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
    plt.xlim([0, len(fX)])
    plt.ylim([0, 30])
    plt.title("10D Levy function")

    plt.tight_layout()
    plt.show()
