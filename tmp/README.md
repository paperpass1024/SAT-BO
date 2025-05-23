# SAT-BO
An Efficient SAT-Based Bayesian Optimization Algorithm for Verification Rules High-Risk Vulnerability Detection

#### Compiler Environmen

Running on linux platform

##### python

see ./requirements.txt

```
cd code
python setup.py install
```

##### c++

./SAT/CMakeLists.txt

```
cmake 3.22	
```

###### build

```
./build.sh
```
#### Operating parameters

##### parameters setting

```
Necessary parameters
--input file //Input CNF Instance
--traffic //Simulation environment path
Non essential parameters
--iteration //Number of SAT and Bayes iterations,default is 6
--solveUppers //The number of solutions recommended by the sat solver to bo each time,default is 30
```

##### Run  example

``` 
cd code 
python main.py --input ../benchmark_verificationRule-SAT-Encoding/v68-c121-1.cnf --traffic ../virtual/binomial/v68-c121-1.txt
```

sat_optimize is our Adaptive DPLL SAT Solver in paper

#### Directory Overview:

```
./code/SAT //SAT solver
./code/turbo //Bayes algorithm
./log/weight //Variable attribute preference file
./log/solve //Solution file used to store SAT solver's output
./ans：//It is the solution file of traffic coverage corresponding to the calculation returned by SAT.
./offline //Three generated traffic environment paths
```
