addpath(genpath('.'))
total_tic = tic;

%initialize global simulation and parts of it
global LBFGS_simulation;
LBFGS_simulation = celes_simulation;
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
initialField.polarization = 'TE';
%0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

%sphere initial condition
parameters = zeros(100,5);
radii = ones(100,3)*500;
radiusArray = radii+200;
parameters(:,1:3) = radiusArray;
parameters(:,5) = parameters(:,5) + refractiveIndex;

%upper and lower bounds
max_rad = 1200*ones(length(radii(:)),1);
min_rad = 500*ones(length(radii(:)),1);
max_param = [max_rad; 20*pi*ones(length(radiusArray(:,1)),1)];
min_param = [min_rad; -20*pi*ones(length(radiusArray(:,1)),1)];

%grid of spheres
xpos = linspace(-33000,33000,10);
ypos = xpos';
zpos = linspace(0,0,1);
[xx,yy,zz] = meshgrid(xpos,ypos,zpos);
positions = zeros(length(radii(:,1)),3);
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
numerics.partitionEdgeSizes = [22000,22000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
% global LBFGS_points;
% global LBFGS_image;
% LBFGS_points = [0,0,5000;-5000,0,5000;5000,0,5000];
% LBFGS_image = [5,5,5];

R = 5000;
x = R*cos(linspace(0,2*pi,12));
y = R*sin(linspace(0,2*pi,12));
z = linspace(10000,100000,12);

image = 12*ones(12,1);
points = zeros(12,3);
points(:,1) = x;
points(:,2) = y;
points(:,3) = z;

LBFGS_image = image;
LBFGS_points = points;

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
LBFGS_simulation.input = input;
LBFGS_simulation.tables.pmax = LBFGS_simulation.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
LBFGS_simulation.numerics = numerics;
LBFGS_simulation.tables = celes_tables;
LBFGS_simulation.output = output;
LBFGS_simulation.tables.nmax = LBFGS_simulation.numerics.nmax;
LBFGS_simulation.tables.pmax = LBFGS_simulation.input.particles.number;
LBFGS_simulation = LBFGS_simulation.computeInitialFieldPower;
LBFGS_simulation = LBFGS_simulation.computeTranslationTable;
LBFGS_simulation.input.particles = LBFGS_simulation.input.particles.compute_maximal_particle_distance;

if strcmp(LBFGS_simulation.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(LBFGS_simulation.input.particles.positionArray,LBFGS_simulation.numerics.partitionEdgeSizes);
    LBFGS_simulation.numerics.partitioning = partitioning;
    LBFGS_simulation.numerics.solver.preconditioner.partitioning = partitioning;
    LBFGS_simulation.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    LBFGS_simulation = LBFGS_simulation.numerics.prepareW(LBFGS_simulation);
    LBFGS_simulation.numerics.solver.preconditioner.partitioningIdcs = LBFGS_simulation.numerics.partitioningIdcs;
    LBFGS_simulation.numerics.file:///home/noise/NOISE_data/Alan/celes-optimization-ellipsoid_angle_cleaninverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation.numerics.partitioningIdcs;
end


celes_func = @(x) CELES_ellipsoid_LBFGS_iteration(x);

initial_params = parameters(:,1:4);

opts = struct( 'x0', initial_params(:),'m',10);
lbfgs_tic = tic;
[r_f, fom_f, info] = lbfgsb(celes_func,min_param,max_param,opts);
lbfgs_time = toc(lbfgs_tic);
total_time = toc(total_tic);


z_i = points(:,3);
field_pts = zeros(10000,3);
[x_i,y_i] = meshgrid(linspace(-10000,10000,100),linspace(-10000,10000,100).');
field_pts(:,1) = x_i(:);
field_pts(:,2) = y_i(:);

for i = 1:length(z_i)
    field_pts(:,3) = z_i(i);
    E = compute_scattered_field_opt(LBFGS_simulation,field_pts);
    I = sum(abs(E).^2,2);
    I = reshape(I,100,100);
    
    figure
    imagesc(I)
    colorbar
end