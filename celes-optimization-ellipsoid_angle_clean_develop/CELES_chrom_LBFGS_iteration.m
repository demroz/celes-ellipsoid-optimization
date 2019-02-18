function [fom, gradient] = CELES_chrom_LBFGS_iteration(r_k)
%%declare global properties
%simulation to be updated per iteration
global LBFGS_simulation_1;
global LBFGS_simulation_2;
%points to calculate figure of merit
global LBFGS_points_1;
global LBFGS_points_2;
%image figure of merit
global LBFGS_image_1;
global LBFGS_image_2;

%input r_k into simulation
LBFGS_simulation_1.input.particles.radiusArray = r_k;
LBFGS_simulation_2.input.particles.radiusArray = r_k;

%%forward simulation
%compute mie coeff
LBFGS_simulation_1 = LBFGS_simulation_1.computeParallelMieCoefficients;
LBFGS_simulation_2 = LBFGS_simulation_2.computeParallelMieCoefficients;

%prepare preconditioner for solver
LBFGS_simulation_1 = LBFGS_simulation_1.numerics.solver.preconditioner.prepareM(LBFGS_simulation_1);
LBFGS_simulation_1 = LBFGS_simulation_1.computeInitialFieldCoefficients;
LBFGS_simulation_2 = LBFGS_simulation_2.numerics.solver.preconditioner.prepareM(LBFGS_simulation_2);
LBFGS_simulation_2 = LBFGS_simulation_2.computeInitialFieldCoefficients;

%solve linear system and clear preconditioner
LBFGS_simulation_1 = LBFGS_simulation_1.computeScatteredFieldCoefficients();
LBFGS_simulation_1.numerics.solver.preconditioner.factorizedMasterMatrices = [];
LBFGS_simulation_2 = LBFGS_simulation_2.computeScatteredFieldCoefficients();
LBFGS_simulation_2.numerics.solver.preconditioner.factorizedMasterMatrices = [];

%compute scattered electric field
E_s_1 = gather(compute_scattered_field_opt(LBFGS_simulation_1,LBFGS_points_1));
%compute scattered electric field
E_s_2 = gather(compute_scattered_field_opt(LBFGS_simulation_2,LBFGS_points_2));

%compute figure of merit
fom = sum((LBFGS_image(:)-sum(abs(gather(E_t)).^2,2)).^2);

%%inverse simulation
%compute grad mie coefficients
LBFGS_simulation_1 = LBFGS_simulation_1.computeParallelGradMieCoefficients;
LBFGS_simulation_2 = LBFGS_simulation_2.computeParallelGradMieCoefficients;

%coupled coefficients calculation
LBFGS_simulation_1 = LBFGS_simulation_1.computeCoupledScatteringCoefficients;
LBFGS_simulation_2 = LBFGS_simulation_2.computeCoupledScatteringCoefficients;

%prepare inverse preconditioner
LBFGS_simulation_1 = LBFGS_simulation_1.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation_1);
LBFGS_simulation_2 = LBFGS_simulation_2.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation_2);

%solve the system
LBFGS_simulation_1 = LBFGS_simulation_1.computeImageAdjointFields(LBFGS_image_1,E_s_1,LBFGS_points_1);
LBFGS_simulation_2 = LBFGS_simulation_2.computeImageAdjointFields(LBFGS_image_2,E_s_2,LBFGS_points_2);

%clear preconditioner
LBFGS_simulation_1.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
grad1 = compute_adjoint_grad_intensity(LBFGS_simulation_1,1);
grad1n = grad1/sqrt(sum(abs(grad1).^2));
LBFGS_simulation_2.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
grad2 = compute_adjoint_grad_intensity(LBFGS_simulation_2,1);
grad2n = grad2/sqrt(sum(abs(grad2).^2));

max_step_size = 1;
gradient = max_step_size*grad1n;