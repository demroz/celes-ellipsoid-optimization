%run forward simulation only to reproduce data from stored mat file. 
%mat file should include
%r_f: particle radii
%positions: particle positions
%points: points where electric field is to be optimized

%wavelength and lmax to be set manually

addpath(genpath('.'))
%load files
load('LBFGS_helix_3um_final.mat');

% initialize global simulation and parts of it
simulation = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

% settings for CUDA code
lmax = 4;
cuda_compile(lmax);

% wavelength of interest
lambda = 3000;
input.wavelength = lambda;
close all

% initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
%TM = x, TE = y
initialField.polarization = 'TM';
% 0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

% refractive index of scatterer
refractiveIndex = 1.47+0.02i;
mediumIndex = 1;

% upper and lower bounds
max_rad = 1100*ones(900,1);
min_rad = 200*ones(900,1);
parameters = ones(length(r_f),2);
parameters(:,1) = r_f;
parameters(:,2) = parameters(:,2)*1.47;

% particle properties
particles.type = 'sphere';
particles.positionArray = positions;
particles.parameterArray = parameters;
input.mediumRefractiveIndex = mediumIndex;

% solver numerics
numerics.lmax = lmax;
numerics.particleDistanceResolution = 1;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

% solver properties
solver.type = 'BiCGStab';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;

% preconditioner properties
preconditioner.type = 'none';
numerics.partitionEdgeSizes = [22000,22000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
simulation.input = input;
simulation.tables.pmax = simulation.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
simulation.numerics = numerics;
simulation.tables = celes_tables;
simulation.output = output;
simulation.tables.nmax = simulation.numerics.nmax;
simulation.tables.pmax = simulation.input.particles.number;
simulation = simulation.computeInitialFieldPower;
simulation = simulation.computeTranslationTable;
simulation.input.particles = simulation.input.particles.compute_maximal_particle_distance;

if strcmp(simulation.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(simulation.input.particles.positionArray,simulation.numerics.partitionEdgeSizes);
    simulation.numerics.partitioning = partitioning;
    simulation.numerics.solver.preconditioner.partitioning = partitioning;
    simulation.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    simulation = simulation.numerics.prepareW(simulation);
    simulation.numerics.solver.preconditioner.partitioningIdcs = simulation.numerics.partitioningIdcs;
    simulation.numerics.inverseSolver.preconditioner.partitioningIdcs = simulation.numerics.partitioningIdcs;
end

simulation = simulation.computeParallelMieCoefficients;
simulation = simulation.numerics.solver.preconditioner.prepareM(simulation);
simulation = simulation.computeInitialFieldCoefficients;
simulation = simulation.computeScatteredFieldCoefficients();
simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];

z_i = points(1:8,3);
% z_i = linspace(100000,300000,20);
field_pts = zeros(160000,3);
[x_i,y_i] = meshgrid(linspace(-40000,40000,400),linspace(-40000,40000,400).');
field_pts(:,1) = x_i(:);
field_pts(:,2) = y_i(:);

I_save = zeros(400,400,length(z_i));

for i = 1:length(z_i)
    field_pts(:,3) = z_i(i);
    E = compute_scattered_field_opt(simulation,field_pts);
    I = sum(abs(E).^2,2);
    I = gather(reshape(I,400,400));
    I_save(:,:,i) = I;
    
    figure
    imagesc(I)
    title(strcat(num2str(floor(z_i(i)/1000)),' nm'))
    colorbar
end

% z_i = 100000;
% field_points = zeros(411^2,3);
% [x_p,y_p,z_p]  = meshgrid(linspace(-41000,41000,411),linspace(-41000,41000,411),z_i);
% field_points(:,1) = x_p(:);
% field_points(:,2) = y_p(:);
% field_points(:,3) = z_p(:);
% 
% E = compute_scattered_field_opt(simulation,field_points);
% I = gather(sum(abs(E).^2,2));
% I = reshape(I,411,411);
% 
% figure
% imagesc(I)
% colorbar
% 
% I_line = I(:,206);
% f = fit(linspace(-41000,41000,411).',I_line,'gauss1');
% fc1 = f.c1;
% sigma = fc1/sqrt(2);
% fwhm = 2*sqrt(2*log(2))*sigma;
%     
% 
[x,z] = meshgrid(linspace(-41000,41000,500),linspace(-1000,300000,1000).'); y=x-x;

output.fieldPoints = [x(:),y(:),z(:)];
output.fieldPointsArrayDims = size(x);
simulation.output = output;
simulation=simulation.evaluateFields;
% figure
% plot_field(gca,simulation,'real Ex','Scattered field',simulation.input.particles.radiusArray,[5000,6000])
% colormap(jet)
% % caxis([0 2])
% colorbar
% figure
% plot_field(gca,simulation,'real Ex','Total field',simulation.input.particles.radiusArray,[5000,6000])
% colormap(jet)
% % caxis([0 2])
% colorbar
% figure
% plot_field(gca,simulation,'real Ey','Scattered field',simulation.input.particles.radiusArray,[5000,6000])
% colormap(jet)
% % caxis([0 2])
% colorbar
% figure
% plot_field(gca,simulation,'real Ey','Total field',simulation.input.particles.radiusArray,[5000,6000])
% colormap(jet)
% % caxis([0 2])
% colorbar
% figure
% plot_field(gca,simulation,'real Ey','Initial field',simulation.input.particles.radiusArray,[5000,6000])
% colormap(jet)
% % caxis([0 2])
% colorbar
figure
plot_field(gca,simulation,'abs E','Total field',simulation.input.particles.parameterArray(:,1),[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_spheres(gca,simulation.input.particles.positionArray,simulation.input.particles.parameterArray(:,1),simulation.input.particles.parameterArray(:,2),'view xy')
axis square