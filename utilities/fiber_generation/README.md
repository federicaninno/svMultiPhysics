# svMultiPhysics fiber generation codes
Python + svMultiPhysics codes for fiber generation. Two methods are implemented:
* Bayer et al. (2012). [link](https://doi.org/10.1007/s10439-012-0593-5)
* Doste et al. (2018). [link](https://doi.org/10.1002/cnm.3185)

## Installation

### Installing as a Python Package
You can install the `fiber_generation` codes as a package using pip:

```bash
pip install -e .
```
This will install all the required packages and will allow you to call the functions in these packages from any directory. 

## Examples
The `main_bayer.py` and `main_doste.py` are scripts to run both methods in the geometry described in the `example/biv_truncated` and `example/biv_with_outflow_tracts` folders respectively.

<img src="example/biv_truncated/bayer_fiber.png" alt="Results for truncated BiV (Bayer)" width="640" />
<img src="example/biv_with_outflow_tracts/doste_fiber.png" alt="Results for BiV w/ outflow tracts (Doste)" width="640" />

Note that the Doste methods needs a geometry with outflow tracts to be run (each valve needs to be defined as a separated surface). Bayer can be run in any biventricular geometry.

For a single ventricle geometry (LV or RV), run `main_bayer.py --single-ventricle` (or set `single_ventricle = True` in the script). Only the epicardium, base and endocardium (`ENDO.vtp`) surfaces are needed. In this variant the helix and transverse angles vary linearly through the wall; see [DOCUMENTATION.md](DOCUMENTATION.md#single-ventricle-variant).


### Documentation
For details ont the implementation and an study of the results, see:
- [DOCUMENTATION.md](DOCUMENTATION.md) - Comprehensive documentation on methods and usage
- [VALIDATION.md](VALIDATION.md) - Validation studies