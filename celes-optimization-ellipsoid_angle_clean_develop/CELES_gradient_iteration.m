function [f, g] = CELES_gradient_iteration(r_k,simulation,points)

simulation.input.particles.parameterArray(:,1) = r_k;


simulation = simulation.computeParallelMieCoefficients;

simulation = simulation.numerics.solver.preconditioner.prepareM(simulation);
simulation = simulation.computeInitialFieldCoefficients;

simulation = simulation.computeScatteredFieldCoefficients;
simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];

E_s = gather(compute_scattered_field_opt(simulation,points));

f = sum(abs(gather(E_s)).^2,2);

simulation = simulation.computeParallelGradMieCoefficients;
simulation = simulation.computeCoupledScatteringCoefficients;

simulation = simulation.numerics.inverseSolver.preconditioner.prepareMt(simulation);

simulation = simulation.computeAdjointCoefficients2(E_s,points);

simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];

g = compute_adjoint_grad_intensity(simulation,1);

