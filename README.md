# AW-PINN for solving 3D flow over a single tall building
Adoptive weights Physics-Informed Neural networks for solving 3D turbulent flow in steady-state condition. 
due to complexity of case only L-BFGS optimization method is considered. A kind of novel adaptive method for weight balance is applied for making the trainer more robust.
L2 regularization in Adam optimization is considered. 

### Caution:
the main file is adoptive_LBGFS_singleCube.py . The data file is created in csv format but there is no restriction for any kind of data for input. vtk or mat can be easily considered.

 ** this code is naïve and needs evolution **



### This case solves 3D flow over a city.the benchmark is Case E in Aij institute:
_Guidebook for CFD Predictions of Urban Wind Environment Architectural Institute of Japan.




### Step 1 - How to run this example
* provide data and boundary condition points file in csv format.
* share your data in your gdrive
* load your drive in colbab
* open the repository github link with google colab
* Run All

### Step 2 - consideration
* Epochs number is up to your case
* it's recommended to run it with GPU 

### Step 3 - Libraries
* torch
* pandas
* numpy
* matplotlib
* sklearn _optional

you can install these libraries in one line easily. just copy this single in in your console:
* pip
  ```sh
  python -m pip install torch pandas numpy matplotlib sklearn
  ```


## Contact

Amirreza Rezayan -  arezayan87@gmail.com

