function [a] = test_global(x)
global lbfgs_simulation;
lbfgs_simulation.output = x;
a = lbfgs_simulation;
end