function [fom, gradient] = CELES_ellipsoid_pol_LBFGS_iteration(r_k)
%%declare global properties
%simulation to be updated per iteration
global LBFGS_simulation_TE;
global LBFGS_simulation_TM;
%points to calculate figure of merit
global LBFGS_points_TE;
global LBFGS_points_TM;
%image figure of merit
global LBFGS_image_TE;
global LBFGS_image_TM;

r_k = reshape(r_k,length(r_k(:))/4,4);

%input r_k into simulation
LBFGS_simulation_TE.input.particles.parameterArray(:,1:4) = r_k;
LBFGS_simulation_TM.input.particles.parameterArray(:,1:4) = r_k;

%%forward simulation
%compute mie coeff
LBFGS_simulation_TE = LBFGS_simulation_TE.computeParallelMieCoefficients;
LBFGS_simulation_TM.tables.singleParticleArrayIndex = LBFGS_simulation_TE.tables.singleParticleArrayIndex;
LBFGS_simulation_TM.tables.mieCoefficients = LBFGS_simulation_TE.tables.mieCoefficients;
LBFGS_simulation_TM.tables.gradMieCoefficients = LBFGS_simulation_TE.tables.gradMieCoefficients;

%prepare preconditioner for solver
LBFGS_simulation_TE = LBFGS_simulation_TE.numerics.solver.preconditioner.prepareM(LBFGS_simulation_TE);
LBFGS_simulation_TM = LBFGS_simulation_TM.numerics.solver.preconditioner.prepareM(LBFGS_simulation_TM);
LBFGS_simulation_TE = LBFGS_simulation_TE.computeInitialFieldCoefficients;
LBFGS_simulation_TM = LBFGS_simulation_TM.computeInitialFieldCoefficients;

%solve linear system and clear preconditioner
LBFGS_simulation_TE = LBFGS_simulation_TE.computeScatteredFieldCoefficients();
LBFGS_simulation_TM = LBFGS_simulation_TM.computeScatteredFieldCoefficients();
LBFGS_simulation_TE.numerics.solver.preconditioner.factorizedMasterMatrices = [];
LBFGS_simulation_TM.numerics.solver.preconditioner.factorizedMasterMatrices = [];

%compute total electric field
E_s_TE = gather(compute_scattered_field_opt(LBFGS_simulation_TE,LBFGS_points_TE));
E_t_TE = E_s_TE;
E_s_TM = gather(compute_scattered_field_opt(LBFGS_simulation_TM,LBFGS_points_TM));
E_t_TM = E_s_TM;

%compute figure of merit
fom = sum((LBFGS_image_TE(:)-sum(abs(gather(E_t_TE)).^2,2)).^2+(LBFGS_image_TM(:)-sum(abs(gather(E_t_TM)).^2,2)).^2);

%%inverse simulation
%compute grad mie coefficients
LBFGS_simulation_TE = LBFGS_simulation_TE.computeParallelGradMieCoefficients;

%coupled coefficients calculation
LBFGS_simulation_TE = LBFGS_simulation_TE.computeCoupledScatteringCoefficients;
LBFGS_simulation_TM = LBFGS_simulation_TM.computeCoupledScatteringCoefficients;

%prepare inverse preconditioner
LBFGS_simulation_TE = LBFGS_simulation_TE.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation_TE);
LBFGS_simulation_TM = LBFGS_simulation_TM.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation_TM);

%solve the system
LBFGS_simulation_TE = LBFGS_simulation_TE.computeImageAdjointCoefficients(LBFGS_image_TE,E_t_TE,LBFGS_points_TE);
LBFGS_simulation_TM = LBFGS_simulation_TM.computeImageAdjointCoefficients(LBFGS_image_TM,E_t_TM,LBFGS_points_TM);

%clear preconditioner
LBFGS_simulation_TE.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
LBFGS_simulation_TM.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
gradTE_1 = compute_adjoint_grad_intensity(LBFGS_simulation_TE,1);
gradTE_2 = compute_adjoint_grad_intensity(LBFGS_simulation_TE,2);
gradTE_3 = compute_adjoint_grad_intensity(LBFGS_simulation_TE,3);
gradTE_4 = compute_adjoint_grad_intensity(LBFGS_simulation_TE,4);
gradTM_1 = compute_adjoint_grad_intensity(LBFGS_simulation_TM,1);
gradTM_2 = compute_adjoint_grad_intensity(LBFGS_simulation_TM,2);
gradTM_3 = compute_adjoint_grad_intensity(LBFGS_simulation_TM,3);
gradTM_4 = compute_adjoint_grad_intensity(LBFGS_simulation_TM,4);

grad = zeros(size(r_k));
grad(:,1) = gradTE_1+gradTM_1;
grad(:,2) = gradTE_2+gradTM_2;
grad(:,3) = gradTE_3+gradTM_3;
grad(:,4) = gradTE_4+gradTM_4;

%grad = grad./max(grad);

max_step_size = 1;
gradient = max_step_size*grad(:);