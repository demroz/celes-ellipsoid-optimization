addpath(genpath('.'))
%load files
load('LBFGS_pol_focus1.mat');

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
lambda = 1550;
input.wavelength = lambda;

% initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
%TM = x, TE = y
initialField.polarization = 'TM';
% 0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

% refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

% upper and lower bounds
max_rad = 1100*ones(900,1);
min_rad = 200*ones(900,1);

% particle properties
particles.type = 'ellipsoid';
particles.refractiveIndexArray = refractiveIndex*ones(1,length(r_f)/3);
particles.positionArray = positions;
particles.radiusArray = reshape(r_f,length(r_f)/3,3);
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
preconditioner.type = 'blockdiagonal';
numerics.partitionEdgeSizes = [16000,16000,16000];
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

% z_i = points(:,3);
% field_pts = zeros(10000,3);
% [x_i,y_i] = meshgrid(linspace(-10000,10000,100),linspace(-10000,10000,100).');
% field_pts(:,1) = x_i(:);
% field_pts(:,2) = y_i(:);
% 
% for i = 1:length(z_i)
%     field_pts(:,3) = z_i(i);
%     E = compute_scattered_field_opt(simulation,field_pts);
%     I = sum(abs(E).^2,2);
%     I = reshape(I,100,100);
%     
%     figure
%     imagesc(I)
%     colorbar
% end

[x,z] = meshgrid(-20000:80:20000,-1000:80:22000); y=x-x;

output.fieldPoints = [x(:),y(:),z(:)];
output.fieldPointsArrayDims = size(x);
simulation.output = output;
simulation=simulation.evaluateFields;
figure
plot_field(gca,simulation,'real Ex','Scattered field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_field(gca,simulation,'real Ex','Total field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_field(gca,simulation,'real Ey','Scattered field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_field(gca,simulation,'real Ey','Total field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_field(gca,simulation,'real Ey','Initial field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_field(gca,simulation,'abs E','Total field',simulation.input.particles.radiusArray,[5000,6000])
colormap(jet)
% caxis([0 2])
colorbar
figure
plot_spheres(gca,simulation.input.particles.positionArray,simulation.input.particles.radiusArray,simulation.input.particles.refractiveIndexArray,'view xy')
%