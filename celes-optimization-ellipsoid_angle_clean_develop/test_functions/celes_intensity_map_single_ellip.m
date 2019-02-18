addpath(genpath('.'))

simulation = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics;
solver = celes_solver;
inverseSolver = celes_solver;
preconditioner = celes_preconditioner;
inversePreconditioner = celes_preconditioner_new;
output = celes_output;

lmax = 4;
cuda_compile(lmax);
cuda_compile_T(lmax);
numPoints = 4;

parameter1 = linspace(0.01,0.02,numPoints);
parameter2 = linspace(0.01,0.02,numPoints)';
axisFlag1 = 4;
axisFlag2 = 4;

initial_params = zeros(1,5);
initial_params(1) = 800;
initial_params(2) = 700;
initial_params(3) = 900;
initial_params(4) = 4*pi/20;
initial_params(5) = 1.47;

point = [0,0,5000];

[parameter1_map, parameter2_map] = meshgrid(parameter1,parameter2);
parameter1_map = parameter1_map(:);
parameter2_map = parameter2_map(:);

positions = zeros(1,3);
particles.positionArray = positions;
particles.type = 'ellipsoid';

% polar angle of incoming beam/wave, in radians (for Gaussian beams, 
% only 0 and pi are currently possible)
initialField.polarAngle = 0;

% azimuthal angle of incoming beam/wave, in radians
initialField.azimuthalAngle = 0;

% polarization of incoming beam/wave ('TE' or 'TM')
initialField.polarization = 'TE';

% width of beam waist (use 0 or inf for plane wave)
initialField.beamWidth = 0;

% focal point 
initialField.focalPoint = [0,0,0];

% vacuum wavelength (same unit as particle positions and radius)
input.wavelength = 1550;

% complex refractive index of surrounding medium
input.mediumRefractiveIndex = 1;

% maximal expansion order of scattered fields (around particle center)
numerics.lmax = lmax;

% resolution of lookup table for spherical Hankel function (same unit as
% wavelength)
numerics.particleDistanceResolution = 1;

% use GPU for various calculations (deactivate if you experience GPU memory 
% problems - translation operator always runs on gpu, even if false)
numerics.gpuFlag = true;

% sampling of polar angles in the plane wave patterns (radians array)
numerics.polarAnglesArray = 0:pi/1e3:pi;

% sampling of azimuthal angles in the plane wave patterns (radians array)
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

% specify solver type (currently 'BiCGStab' or 'GMRES')
solver.type = 'BiCGStab';

% relative accuracy (solver stops when achieved)
solver.tolerance=1e-4;

% maximal number of iterations (solver stops if exceeded)
solver.maxIter=1000;

% restart parameter (only for GMRES)
solver.restart=1000;

% type of preconditioner (currently only 'blockdiagonal' and 'none'
% possible)
preconditioner.type = 'none';
inversePreconditioner.type = 'none';
inversePreconditioner.partitionEdgeSizes = [3000,3000,3000];
% for blockdiagonal preconditioner: edge size of partitioning cuboids
preconditioner.partitionEdgeSizes = [1200,1200,1200];

input.initialField = initialField;
solver.preconditioner = preconditioner; 
inverseSolver.preconditioner = inversePreconditioner;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
simulation.numerics = numerics;
simulation.tables = celes_tables;
simulation.output = output;
simulation.tables.nmax = simulation.numerics.nmax;

intensities = zeros(numPoints^2,1);
gradients = zeros(numPoints^2,2);
gradients1 = gradients;
timingadj = ones(numPoints^2,1);
timingdum = timingadj;

for i = 1:numPoints^2
    parameter1_i = parameter1_map(i);
    parameter2_i = parameter2_map(i);
    ellip_param = initial_params;
    ellip_param(axisFlag1) = parameter1_i;
    ellip_param(axisFlag2) = parameter2_i;
    particles.parameterArray = ellip_param;
    input.particles = particles;
    simulation.input = input;
    simulation.tables.pmax = simulation.input.particles.number;
    simulation = simulation.computeInitialFieldPower;
    simulation = simulation.computeTranslationTable;
    simulation.input.particles = simulation.input.particles.compute_maximal_particle_distance;


    %compute mie coefficients
    simulation = simulation.computeParallelMieCoefficients;
    
    %find initial field coefficients corresponding to input beam
    simulation = simulation.computeInitialFieldCoefficients;
    
    %solve for scattered field coefficients
    simulation = simulation.computeScatteredFieldCoefficients();
    
    E_t = compute_scattered_field_opt(simulation,point);%+compute_initial_field_opt(simulation,point);
    intensities(i) = sum(abs(gather(E_t))).^2;
    simulation=simulation.computeGradMieCoefficients;
    simulation=simulation.computeCoupledScatteringCoefficients;
%     simulation=simulation.computeGradScatteredFieldCoefficients;
%     %compute gradient of E for gradient figure of merit
% %     grad_E = compute_electric_field_gradient(simulation,point);
% %     E = conj(compute_scattered_field_opt(simulation,point));
% %     grad = 2*real(bsxfun(@times,grad_E',E));
% %     grad = sum(grad');
%     grad = compute_intensity_gradient(simulation,point);
% %     timingdum(i) = toc;
    tic;
    simulation=simulation.computeAdjointCoefficients(E_t,point);
    grad1 = compute_adjoint_grad_intensity(simulation,axisFlag1);
    grad2 = compute_adjoint_grad_intensity(simulation,axisFlag2);
    timingadj(i) = toc;
%     %reformulate gradient
%     raw_grad = squeeze(gather(grad))';
% %     gradients(i,:) = raw_grad;
    gradients1(i,:) = [grad1,grad2];
end

intensities = reshape(intensities,[numPoints,numPoints]);
% figure
% imagesc(particle1,particle1,intensities);
% colorbar;

% contour_pts = old_stored_radius(:);
% contour_pts = sort(contour_pts);
% contour_pts = contour_pts(1:10:end);
% figure
% contour(particle1,particle1,intensities,50);
% hold on
% quiver(stored_radius(:,1),stored_radius(:,2),stored_gradient(:,1),stored_gradient(:,2));
% axis square

% figure
% surf(intensities);
% 
% figure
% plot(timingdum,'r-');
% hold
% plot(timingadj,'b');


[dx,dy] = gradient(intensities);
% d = gradient(intensities);

% figure
% contour(particle1,particle1,intensities,200);
% hold on
% quiver(particle1,particle1,dx,dy);
% hold on
% quiver(stored_radius(:,1),stored_radius(:,2),stored_gradient(:,1),stored_gradient(:,2),'r');
% hold on
% quiver(old_stored_radius(:,1),old_stored_radius(:,2),old_stored_gradient(:,1),old_stored_gradient(:,2),'r');
% axis square

% figure
% plot(particle1,intensities);
% 
% figure
% plot(particle1,gradients/max(abs(gradients)));
% hold on
% plot(particle1,d/max(abs(d)));
% 
% figure
% contour(particle1,particle2,intensities,30);
% % hold on
% % quiver(radius_map1,radius_map2,gradients(:,1),gradients(:,2));
% hold on
% quiver(particle1,particle2,dx,dy,'r');
% axis square
% % quiver(radius_map1,radius_map2,gradients1(:,1),gradients1(:,2));
% 
% figure
% contour(particle1,particle2,intensities,30);
% % hold on
% % quiver(radius_map1,radius_map2,gradients(:,1),gradients(:,2));
% hold on
% % quiver(particle1,particle2,dx,dy,'r');
% axis square
% quiver(radius_map1,radius_map2,gradients1(:,1),gradients1(:,2));

figure
contour(parameter1,parameter2,intensities,30);
% hold on
% quiver(radius_map1,radius_map2,gradients(:,1),gradients(:,2));
hold on
quiver(parameter1,parameter2,dx,dy,'r');
hold on
% quiver(particle1,particle2,dx,dy,'r');
quiver(parameter1_map,parameter2_map,gradients1(:,1),gradients1(:,2));
axis square

% quiver(stored_radius(:,1),stored_radius(:,2),stored_gradient(:,1),stored_gradient(:,2),'r');
% hold on
% quiver(old_stored_radius(:,1),old_stored_radius(:,2),old_stored_gradient(:,1),old_stored_gradient(:,2),'r');
% axis square
% 
% a = diag(intensities);
% figure
% plot(particle1,a)

% save('intensity_grad_map_lmax2_r100-500.mat','particle1','particle2','intensities','radius_map1','radius_map2','gradients');
normd = sqrt(dx(:).^2+dy(:).^2);
normdx = dx(:)./normd;
normdy = dy(:)./normd;

normg = sqrt(sum(gradients1.^2,2));
normgx = gradients1(:,1)./normg;
normgy = gradients1(:,2)./normg;

dgx = normgx-normdx;
dgy = normgy-normdy;

figure
plot(dgx)
hold on
plot(dgy)

figure
subplot(2,1,1)
imagesc(reshape(dgx.',numPoints,numPoints));
colorbar
title('diff x')
subplot(2,1,2)
imagesc(reshape(dgy.',numPoints,numPoints));
colorbar
title('diff y')

abs_dg = sqrt(normdx.^2+normdy.^2).*sqrt(normgx.^2+normgy.^2);
dot_dg = normdx.*normgx+normdy.*normgy;

angle_dg = acos(dot_dg./abs_dg);

figure
imagesc(reshape(angle_dg,numPoints,numPoints))
colorbar