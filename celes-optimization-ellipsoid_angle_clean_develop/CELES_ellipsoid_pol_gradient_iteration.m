function [fom, gradient] = CELES_ellipsoid_pol_gradient_iteration(r_k,simulation_TE,points_TE,image_TE,simulation_TM,points_TM,image_TM)

r_k = reshape(r_k,length(r_k(:))/4,4);

%input r_k into simulation
simulation_TE.input.particles.parameterArray(:,1:4) = r_k;
simulation_TM.input.particles.parameterArray(:,1:4) = r_k;

%%forward simulation
%compute mie coeff
coeff_start = tic;
simulation_TE = simulation_TE.computeParallelMieCoefficients;
coeff_time = toc(coeff_start);
fprintf('\n' + string(coeff_time) + 's to compute Mie Coefficients for ' + string(simulation_TE.input.particles.numUniqueParticles) +' unique particles \n');
simulation_TM.tables.singleParticleArrayIndex = simulation_TE.tables.singleParticleArrayIndex;
simulation_TM.tables.mieCoefficients = simulation_TE.tables.mieCoefficients;
simulation_TM.tables.gradMieCoefficients = simulation_TE.tables.gradMieCoefficients;

%prepare preconditioner for solver
precond_start = tic;
simulation_TE = simulation_TE.numerics.solver.preconditioner.prepareM(simulation_TE);
simulation_TM = simulation_TM.numerics.solver.preconditioner.prepareM(simulation_TM);
precond_time = toc(precond_start);
fprintf('\n' + string(precond_time) + 's to factorize TE and TM forward precond \n');
simulation_TE = simulation_TE.computeInitialFieldCoefficients;
simulation_TM = simulation_TM.computeInitialFieldCoefficients;

%solve linear system and clear preconditioner
forward_start = tic;
simulation_TE = simulation_TE.computeScatteredFieldCoefficients();
simulation_TM = simulation_TM.computeScatteredFieldCoefficients();
forward_time = toc(forward_start);
fprintf('\n' + string(forward_time) + 's to solve forward problem \n');
simulation_TE.numerics.solver.preconditioner.factorizedMasterMatrices = [];
simulation_TM.numerics.solver.preconditioner.factorizedMasterMatrices = [];

%compute total electric field
field_start = tic;
E_s_TE = gather(compute_scattered_field_opt(simulation_TE,points_TE));
E_t_TE = E_s_TE;
E_s_TM = gather(compute_scattered_field_opt(simulation_TM,points_TM));
E_t_TM = E_s_TM;
field_time = toc(field_start);
fprintf('\n' + string(field_time) + 's to calculate fields \n');

%compute figure of merit
fom = sum((image_TE(:)-sum(abs(gather(E_t_TE)).^2,2)).^2+(image_TM(:)-sum(abs(gather(E_t_TM)).^2,2)).^2);

%%inverse simulation
%compute grad mie coefficients
simulation_TE = simulation_TE.computeParallelGradMieCoefficients;

%coupled coefficients calculation
simulation_TE = simulation_TE.computeCoupledScatteringCoefficients;
simulation_TM = simulation_TM.computeCoupledScatteringCoefficients;

%prepare inverse preconditioner
precond_start = tic;
simulation_TE = simulation_TE.numerics.inverseSolver.preconditioner.prepareMt(simulation_TE);
simulation_TM = simulation_TM.numerics.inverseSolver.preconditioner.prepareMt(simulation_TM);
precond_time = toc(precond_start);
fprintf('\n' + string(precond_time) + 's to factorize TE and TM adjoint precond \n');

%solve the system
adjoint_start = tic;
simulation_TE = simulation_TE.computeImageAdjointCoefficients(image_TE,E_t_TE,points_TE);
simulation_TM = simulation_TM.computeImageAdjointCoefficients(image_TM,E_t_TM,points_TM);
adjoint_time = toc(adjoint_start);
fprintf('\n' + string(adjoint_time) + 's to solve adjoint problem \n');

%clear preconditioner
simulation_TE.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
simulation_TM.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
gradTE_1 = compute_adjoint_grad_intensity(simulation_TE,1);
gradTE_2 = compute_adjoint_grad_intensity(simulation_TE,2);
gradTE_3 = compute_adjoint_grad_intensity(simulation_TE,3);
gradTE_4 = compute_adjoint_grad_intensity(simulation_TE,4);
gradTM_1 = compute_adjoint_grad_intensity(simulation_TM,1);
gradTM_2 = compute_adjoint_grad_intensity(simulation_TM,2);
gradTM_3 = compute_adjoint_grad_intensity(simulation_TM,3);
gradTM_4 = compute_adjoint_grad_intensity(simulation_TM,4);

grad = zeros(size(r_k));
grad(:,1) = gradTE_1+gradTM_1;
grad(:,2) = gradTE_2+gradTM_2;
grad(:,3) = gradTE_3+gradTM_3;
grad(:,4) = gradTE_4+gradTM_4;

%grad = grad./max(grad);

max_step_size = 1;
gradient = max_step_size*grad(:);