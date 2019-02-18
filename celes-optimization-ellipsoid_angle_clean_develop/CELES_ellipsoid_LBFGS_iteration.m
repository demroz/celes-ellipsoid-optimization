function [fom, gradient] = CELES_ellipsoid_LBFGS_iteration(r_k)
%%declare global properties
%simulation to be updated per iteration
global LBFGS_simulation;
%points to calculate figure of merit
global LBFGS_points;
%image figure of merit
global LBFGS_image;

r_k = reshape(r_k,length(r_k(:))/4,4);

%input r_k into simulation
LBFGS_simulation.input.particles.parameterArray(:,1:4) = r_k;

%%forward simulation
%compute mie coeff
LBFGS_simulation = LBFGS_simulation.computeParallelMieCoefficients;

%prepare preconditioner for solver
LBFGS_simulation = LBFGS_simulation.numerics.solver.preconditioner.prepareM(LBFGS_simulation);
LBFGS_simulation = LBFGS_simulation.computeInitialFieldCoefficients;

%solve linear system and clear preconditioner
LBFGS_simulation = LBFGS_simulation.computeScatteredFieldCoefficients();
LBFGS_simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];

%compute total electric field
E_s = gather(compute_scattered_field_opt(LBFGS_simulation,LBFGS_points));
E_t = E_s;

%compute figure of merit
fom = sum((LBFGS_image(:)-sum(abs(gather(E_t)).^2,2)).^2);

%%inverse simulation
%compute grad mie coefficients
LBFGS_simulation = LBFGS_simulation.computeParallelGradMieCoefficients;

%coupled coefficients calculation
LBFGS_simulation = LBFGS_simulation.computeCoupledScatteringCoefficients;

%prepare inverse preconditioner
LBFGS_simulation = LBFGS_simulation.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation);

%solve the system
LBFGS_simulation = LBFGS_simulation.computeImageAdjointFields(LBFGS_image,E_t,LBFGS_points);

%clear preconditioner
LBFGS_simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
grad1 = compute_adjoint_grad_intensity(LBFGS_simulation,1);
grad2 = compute_adjoint_grad_intensity(LBFGS_simulation,2);
grad3 = compute_adjoint_grad_intensity(LBFGS_simulation,3);
grad4 = compute_adjoint_grad_intensity(LBFGS_simulation,4);

grad = zeros(size(r_k));
grad(:,1) = grad1;
grad(:,2) = grad2;
grad(:,3) = grad3;
grad(:,4) = grad4;

% grad = grad/sqrt(sum((grad(:)).^2));

max_step_size = 1;
gradient = max_step_size*grad(:);