# The Regression Tsetlin Machine

The inner inference mechanism of the Tsetlin Machine (https://arxiv.org/abs/1804.01508) is modified so that input patterns are transformed into a single continuous output, rather than to distinct categories.

This is achieved by: 

* Using the conjunctive clauses of the Tsetlin Machine to capture arbitrarily complex patterns;
* Mapping these patterns to a continuous output through a novel voting and normalization mechanism; and 
* Employing a feedback scheme that updates the Tsetlin Machine clauses to minimize the regression error. 

## Behaviour with noisy and noise-free data

Six datasets have been given in order to study the behaviour of the Regression Tsetlin Machine.

* **Dataset I** contains  2-bit  feature  input  and  the  output  is  100  times  larger  than  the  decimal value of the binary input (e.g., when the input is [1, 0], the output is 200). The training set consists of 8000 samples while testing set consists of 2000 samples, both without noise
* **Dataset II** contains the same data as Dataset I, except that the output of the training data is perturbed to introduce noise
* **Dataset III** has 3-bit input without noise 
* **Dataset IV** has 3-bit input with noisy output
* **Dataset V** has 4-bit input without noise
* **Dataset VI** has 4-bit input with noise

Different datasets can be called by changing the following line in **_ArtificialDataDemo.py_**
```
df = np.loadtxt("2inNoNoise.txt").astype(dtype=np.float32)
```
<img src="https://github.com/cair/regression-tsetlin-machine/blob/master/Training.PNG" width="450" height="450">


Some Results as an example:
```
Average Absolute Error on Training Data: 0.0
Average Absolute Error on test data: 0.0
```
