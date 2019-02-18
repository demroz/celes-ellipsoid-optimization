addpath(genpath('.'))

%initialize global simulation and parts of it for both wavelengths
global LBFGS_simulation_1;
LBFGS_simulation_1 = celes_simulation;
particles = celes_particles;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

global LBFGS_simulation_2
LBFGS_simulation_2 = celes_simulation;
particles_2 = celes_particles;
initialField_2 = celes_initialField;
input_2 = celes_input;

%settings for CUDA code
lmax = 4;
cuda_compile(lmax);
cuda_compile_T(lmax);

%wavelengths of interest
lambda = 1550;
input.wavelength = lambda;
lambda_2 = 1310;
input_2.wavelength = lambda_2;

%fom
global LBFGS_points_1;
global LBFGS_image_1;
xl = linspace(-10000,10000,50);
yl = xl.';
[xp, yp] = meshgrid(xl,yl);
zp = zeros(50)+20000;
r = sqrt(xp.^2+yp.^2);
r = r./max(max(r));
image = r.*exp(-10*r);
image = image/max(max(image));

LBFGS_points_1 = zeros(2500,3);
LBFGS_points_1(:,1) = xp(:);
LBFGS_points_1(:,2) = yp(:);
LBFGS_points_1(:,3) = zp(:);
LBFGS_image_1 = image;

figure
imagesc(image)
colorbar
title('image 1')

global LBFGS_points_2;
global LBFGS_image_2;
xl = linspace(-10000,10000,50);
yl = xl.';
[xp, yp] = meshgrid(xl,yl);
r = sqrt(xp.^2+yp.^2);
r = r./max(max(r));
zp = zeros(50)+20000;
image_2 = abs(sin(10*r)./r/10).^2;

figure
imagesc(image_2)
colorbar
title('image 2')

LBFGS_points_2 = zeros(2500,3);
LBFGS_points_2(:,1) = xp(:);
LBFGS_points_2(:,2) = yp(:);
LBFGS_points_2(:,3) = zp(:);
LBFGS_image_2 = image;

%initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
%TM = x, TE = y
initialField.polarization = 'TM';
%0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];
initialField_2.polarAngle = 0;
initialField_2.azimuthalAngle = 0;
%TM = x, TE = y
initialField_2.polarization = 'TM';
%0 or inf for plane wave
initialField_2.beamWidth = 0;
initialField_2.focalPoint = [0,0,0];

%refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

refractiveIndex_2 = 1.52;
mediumIndex_2 = 1;

%sphere initial condition
radii = ones(1600,1)*500;
radiusArray = radii+200;

%upper and lower bounds
max_rad = 1200*ones(length(radii(:)),1);
min_rad = 500*ones(length(radii(:)),1);

%grid of spheres
xpos = linspace(-48000,48000,40);
ypos = xpos';
zpos = linspace(0,0,1);
[xx,yy,zz] = meshgrid(xpos,ypos,zpos);
positions = zeros(length(radii(:,1)),3);
positions(:,1) = [xx(:)];
positions(:,2) = [yy(:)];
positions(:,3) = [zz(:)];

%particle properties
particles.type = 'sphere';
particles.refractiveIndexArray = refractiveIndex*ones(1,length(radii(:,1)));
particles.positionArray = positions;
input.mediumRefractiveIndex = mediumIndex;

particles_2.type = 'sphere';
particles_2.refractiveIndexArray = refractiveIndex_2*ones(1,length(radii(:,1)));
particles_2.positionArray = positions;
input_2.mediumRefractiveIndex = mediumIndex_2;

%solver numerics
numerics.lmax = lmax;
numerics.particleDistanceResolution = 0.5;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

%solver properties
solver.type = 'BiCGStab';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;

%preconditioner properties
preconditioner.type = 'blockdiagonal';
numerics.partitionEdgeSizes = [30000,30000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
LBFGS_simulation_1.input = input;
LBFGS_simulation_1.tables.pmax = LBFGS_simulation_1.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
LBFGS_simulation_1.numerics = numerics;
LBFGS_simulation_1.tables = celes_tables;
LBFGS_simulation_1.output = output;
LBFGS_simulation_1.tables.nmax = LBFGS_simulation_1.numerics.nmax;
LBFGS_simulation_1.tables.pmax = LBFGS_simulation_1.input.particles.number;
LBFGS_simulation_1 = LBFGS_simulation_1.computeInitialFieldPower;
LBFGS_simulation_1 = LBFGS_simulation_1.computeTranslationTable;
LBFGS_simulation_1.input.particles = LBFGS_simulation_1.input.particles.compute_maximal_particle_distance;

input_2.intialField = initialField_2;
input_2.particles = particles_2;
LBFGS_simulation_2.input = input_2;
LBFGS_simulation_2.tables.pmax = LBFGS_simulation_2.input.particles.number;
LBFGS_simulation_2.numerics = numerics;
LBFGS_simulation_2.tables = celes_tables;
LBFGS_simulation_2.output = output_2;
LBFGS_simulation_2.tables.nmax = LBFGS_simulation_2.numerics.nmax;
LBFGS_simulation_2.tables.pmax = LBFGS_simulation_2.input.particles.number;
LBFGS_simulation_2 = LBFGS_simulation_2.computeInitialFieldPower;
LBFGS_simulation_2 = LBFGS_simulation_2.computeTranslationTable;
LBFGS_simulation_2.input.particles = LBFGS_simulation_2.input.particles.compute_maximal_particle_distance;


if strcmp(LBFGS_simulation_1.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(LBFGS_simulation_1.input.particles.positionArray,LBFGS_simulation_1.numerics.partitionEdgeSizes);
    LBFGS_simulation_1.numerics.partitioning = partitioning;
    LBFGS_simulation_1.numerics.solver.preconditioner.partitioning = partitioning;
    LBFGS_simulation_1.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    LBFGS_simulation_1 = LBFGS_simulation_1.numerics.prepareW(LBFGS_simulation_1);
    LBFGS_simulation_1.numerics.solver.preconditioner.partitioningIdcs = LBFGS_simulation_1.numerics.partitioningIdcs;
    LBFGS_simulation_1.numerics.inverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation_1.numerics.partitioningIdcs;
end

if strcmp(LBFGS_simulation_2.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(LBFGS_simulation_2.input.particles.positionArray,LBFGS_simulation_2.numerics.partitionEdgeSizes);
    LBFGS_simulation_2.numerics.partitioning = partitioning;
    LBFGS_simulation_2.numerics.solver.preconditioner.partitioning = partitioning;
    LBFGS_simulation_2.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    LBFGS_simulation_2 = LBFGS_simulation_2.numerics.prepareW(LBFGS_simulation_2);
    LBFGS_simulation_2.numerics.solver.preconditioner.partitioningIdcs = LBFGS_simulation_2.numerics.partitioningIdcs;
    LBFGS_simulation_2.numerics.inverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation_2.numerics.partitioningIdcs;
end

celes_func = @(x) CELES_chrom_LBFGS_iteration(x);

opts = struct( 'x0', radii(:));

[r_f, fom_f, info] = lbfgsb(celes_func,min_rad,max_rad,opts);