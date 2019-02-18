addpath(genpath('.'))
total_tic = tic;

%initialize global simulation and parts of it
global LBFGS_simulation_TM;
LBFGS_simulation_TM = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

%settings for CUDA code
lmax = 4;
cuda_compile(lmax);
cuda_compile_T(lmax);

%wavelength of interest
lambda = 1550;
input.wavelength = lambda;

%initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
%TM = x, TE = y for normal incidence
initialField.polarization = 'TM';
%0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

%sphere initial condition
parameters = zeros(1600,5);
radiusArray = 300*rand(1600,3)+600;
parameters(:,1:3) = radiusArray;
parameters(:,5) = parameters(:,5) + refractiveIndex;

%upper and lower bounds
max_rad = 1100*ones(length(radiusArray(:)),1);
min_rad = 500*ones(length(radiusArray(:)),1);
max_param = [max_rad; 20*pi*ones(length(radiusArray(:,1)),1)];
min_param = [min_rad; -20*pi*ones(length(radiusArray(:,1)),1)];

%grid of spheres
xpos = linspace(-45000,45000,40);
ypos = xpos';
zpos = linspace(0,0,1);
[xx,yy,zz] = meshgrid(xpos,ypos,zpos);
positions = zeros(length(radiusArray(:,1)),3);
positions(:,1) = [xx(:)];
positions(:,2) = [yy(:)];
positions(:,3) = [zz(:)];

%particle properties
particles.type = 'ellipsoid';
particles.parameterArray = parameters;
particles.positionArray = positions;
input.mediumRefractiveIndex = mediumIndex;

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
numerics.partitionEdgeSizes = [25000,25000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
global LBFGS_points_TE;
global LBFGS_image_TE;
LBFGS_points_TE = [0,0,15000;0,0,20000];
LBFGS_image_TE = [25,0];

global LBFGS_points_TM;
global LBFGS_image_TM;
LBFGS_points_TM = [0,0,15000;0,0,20000];
LBFGS_image_TM = [0,25];

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
LBFGS_simulation_TM.input = input;
LBFGS_simulation_TM.tables.pmax = LBFGS_simulation_TM.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
LBFGS_simulation_TM.numerics = numerics;
LBFGS_simulation_TM.tables = celes_tables;
LBFGS_simulation_TM.output = output;
LBFGS_simulation_TM.tables.nmax = LBFGS_simulation_TM.numerics.nmax;
LBFGS_simulation_TM.tables.pmax = LBFGS_simulation_TM.input.particles.number;
LBFGS_simulation_TM.tables.particleType = LBFGS_simulation_TM.input.particles.type;
LBFGS_simulation_TM = LBFGS_simulation_TM.computeInitialFieldPower;
LBFGS_simulation_TM = LBFGS_simulation_TM.computeTranslationTable;
LBFGS_simulation_TM.input.particles = LBFGS_simulation_TM.input.particles.compute_maximal_particle_distance;

if strcmp(LBFGS_simulation_TM.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(LBFGS_simulation_TM.input.particles.positionArray,LBFGS_simulation_TM.numerics.partitionEdgeSizes);
    LBFGS_simulation_TM.numerics.partitioning = partitioning;
    LBFGS_simulation_TM.numerics.solver.preconditioner.partitioning = partitioning;
    LBFGS_simulation_TM.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    LBFGS_simulation_TM = LBFGS_simulation_TM.numerics.prepareW(LBFGS_simulation_TM);
    LBFGS_simulation_TM.numerics.solver.preconditioner.partitioningIdcs = LBFGS_simulation_TM.numerics.partitioningIdcs;
    LBFGS_simulation_TM.numerics.inverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation_TM.numerics.partitioningIdcs;
end

global LBFGS_simulation_TE
LBFGS_simulation_TE = LBFGS_simulation_TM;
LBFGS_simulation_TE.input.initialField.polarization = 'TE';

celes_func = @(x) CELES_ellipsoid_pol_LBFGS_iteration(x);

initial_params = parameters(:,1:4);

opts = struct( 'x0', initial_params(:),'m',10,'maxIts',40,'maxTotalIts',150);
lbfgs_tic = tic;
[r_f, fom_f, info] = lbfgsb(celes_func,min_param,max_param,opts);
lbfgs_time = toc(lbfgs_tic);
total_time = toc(total_tic);


% z_i = points(:,3);
% field_pts = zeros(10000,3);
% [x_i,y_i] = meshgrid(linspace(-10000,10000,100),linspace(-10000,10000,100).');
% field_pts(:,1) = x_i(:);
% field_pts(:,2) = y_i(:);
% 
% for i = 1:length(z_i)
%     field_pts(:,3) = z_i(i);
%     E = compute_scattered_field_opt(LBFGS_simulation_TM,field_pts);
%     I = sum(abs(E).^2,2);
%     I = reshape(I,100,100);
%     
%     figure
%     imagesc(I)
%     colorbar
% end