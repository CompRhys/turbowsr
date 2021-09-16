from pymatgen.core.structure import Structure
import matplotlib
import matplotlib.pyplot as plt

from megnet.utils.models import load_model

from turbowsr.opt import TuRBOWSR
from turbo import Turbo1

# s = Structure.from_file("/home/reag2/PhD/turbowsr/turbowsr/tests/POSCAR.mp-554710_AgAsC4S8(N2F3)2")
s = Structure.from_file(
    "/home/reag2/PhD/turbowsr/turbowsr/tests/POSCAR.tricky_symmetry"
)
# s = Structure.from_file("/home/reag2/PhD/turbowsr/turbowsr/tests/POSCAR.mp-862972_AcAgAu2")

model = load_model("Eform_MP_2019")

f = TuRBOWSR(s, model)

print(f.proto)
print(f.chemsys)

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals=100,  # Maximum number of evaluations
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

print("Best value found:\n\tf(x) = {f_best:.3f}")
print(dict(zip(f.params.keys(), x_best)))
print(f.params)

fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({"font.size": 16})
plt.plot(fX, "b.", ms=10)  # Plot all evaluated points as blue dots
plt.plot(np.minimum.accumulate(fX), "r", lw=3)  # Plot cumulative minimum as a red line
plt.xlim([0, len(fX)])

plt.tight_layout()
plt.show()
