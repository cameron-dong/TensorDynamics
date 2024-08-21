# TensorDynamics

TensorDynamics is an atmospheric dynamical core implemented in Tensorflow for usage with GPU's. It solves the primitive equations on sigma levels using Eulerian spherical harmonic transforms. Execution time is approximately 1.5 seconds per model day on an Nvidia 3060 GPU at T85 horizontal resolution with 25 vertical levels.

Examples: baroclinic_instability.ipynb, ERA5_forecasting.ipynb


TODO: Held-Suarez test case. Physical parameterizations for convection and radiation. Test autodifferentiation.


Model formulation is based on relevant section in the following sources:

[1] Durran, D.R., 2010. Numerical methods for fluid dynamics: With applications to geophysics (Vol. 32). Springer Science & Business Media.

[2] Neale, R.B., Chen, C.C., Gettelman, A., Lauritzen, P.H., Park, S., Williamson, D.L., Conley, A.J., Garcia, R., Kinnison, D., Lamarque, J.F. and Marsh, D., 2010. Description of the NCAR community atmosphere model (CAM 5.0). NCAR Tech. Note Ncar/tn-486+ STR, 1(1), pp.1-12.

