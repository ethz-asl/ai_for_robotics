# Ex3 - Solutions

Weighted as follows: I.1 (0.4), I.2 (0.1), II.1 (0.4), II.2 (0.1)

### I.1
Score computed based on RMSE (linear): 
```
RMSE(student) >= RMSE(odometry): score = 0.0
RMSE(student) <= RMSE(reference_optimizer): score = 100
```

### I.2
Score computed based on RMSE (linear): 
```
RMSE(student) >= RMSE(odometry): score = 0.0
RMSE(student) <= RMSE(reference_optimizer_olsen): score = 100
```
The reference solution is computed using [1] but many different approaches lead to good results.

[1]: E. Olsen et al., "Fast Iterative Alignment of Pose Graphs with Poor Initial Estimates", 2006 

### II.1
Score computed based on transformation (translation and rotation) error (linear):
```
error(student) >= error(initial_transformation): score = 0.0
error(student) = error(ground_truth): score = 100.0
```

### II.2
Score computed based on transformation (translation and rotation) error (linear):
```
error(student) >= (7.0): score = 0.0
error(student) <= (2.9): score = 100.0
```
The reference solution is computed using ICP but many different approaches lead to good results.
A relatively high error was tolerated since there is no unique point-to-point mapping.

