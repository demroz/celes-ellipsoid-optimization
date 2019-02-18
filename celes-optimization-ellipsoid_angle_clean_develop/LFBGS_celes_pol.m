addpath(genpath('.'))

%initialize global simulation and parts of it
global LBFGS_simulation;
LBFGS_simulation = celes_simulation;
particles = celes_particles;
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
%TM = x, TE = y
initialField.polarization = 'TM';
%0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

%sphere initial condition
radii = ones(3600,1)*500;
radiusArray = radii+200;

%upper and lower bounds
max_rad = 1200*ones(length(radii(:)),1);
min_rad = 500*ones(length(radii(:)),1);

%grid of spheres
xpos = linspace(-72000,72000,60);
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

%solver numerics
numerics.lmax = lmax;
numerics.particleDistanceResolution = 0.5;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

%solver properties
solver.type = 'GMRES';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;

%preconditioner properties
preconditioner.type = 'blockdiagonal';
numerics.partitionEdgeSizes = [40000,40000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
global LBFGS_points;
global LBFGS_image;
% xl = linspace(-5000,5000,20);
% yl = xl.';
% [xp, yp] = meshgrid(xl,yl);
% zp = zeros(20)+20000;
% r = sqrt(xp.^2+yp.^2);
% r = r./max(max(r));
% image = (r.^2.*exp(-5*r.^2));
% image = image/max(max(image));
% imagePower = sum(sum(image));

R = 10000;
x = R*cos(linspace(0,2*pi,20));
y = R*sin(linspace(0,2*pi,20));
z = linspace(100000,150000,20);

image = 20*ones(20,1);
points = zeros(20,3);
points(:,1) = x;
points(:,2) = y;
points(:,3) = z;

% %calculate normalization factor
% differences = diff(xl);
% dx = differences(1);
% xi = xpos(1):dx:xpos(end);
% initialPower = length(xi)^2;

% %define fraction of initial power wanted in focus
% %roughly efficiency
% powerFract = 0.1;
% 
% %normalize image power to powerFract and initialPower
% image = powerFract*initialPower/imagePower * image;

% %image = [10,10,10];
% points = zeros(length(xp(:)),3);
% 
% points(:,1) = xp(:);
% points(:,2) = yp(:);
% points(:,3) = zp(:);

figure
imagesc(image)
colorbar
LBFGS_points = points;
LBFGS_image = image;

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
    LBFGS_simulation.numerics.inverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation.numerics.partitioningIdcs;
end


celes_func = @(x) CELES_LBFGS_iteration(x);

opts = struct( 'x0', radii(:),'m',10);

tic
[r_f, fom_f, info] = lbfgsb(celes_func,min_rad,max_rad,opts);
lbfgs_time = toc;