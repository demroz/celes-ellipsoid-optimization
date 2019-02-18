addpath(genpath('.'))
total_tic = tic;

%initialize global simulation and parts of it
simulation_TM = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

totalIterations = 100;
maxStepSize = 10;

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
radiusArray = 700*ones(1600,3);
parameters(:,1:3) = radiusArray;
parameters(:,5) = parameters(:,5) + refractiveIndex;

%upper and lower bounds
maxAxis = 1100;
minAxis = 500;

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
numerics.partitionEdgeSizes = [35000,35000,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
points_TE = [0,0,20000;0,0,30000];
image_TE = [55,0];

points_TM = [0,0,20000;0,0,30000];
image_TM = [0,55];

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
simulation_TM.input = input;
simulation_TM.tables.pmax = simulation_TM.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
simulation_TM.numerics = numerics;
simulation_TM.tables = celes_tables;
simulation_TM.output = output;
simulation_TM.tables.nmax = simulation_TM.numerics.nmax;
simulation_TM.tables.pmax = simulation_TM.input.particles.number;
simulation_TM.tables.particleType = simulation_TM.input.particles.type;
simulation_TM = simulation_TM.computeInitialFieldPower;
simulation_TM = simulation_TM.computeTranslationTable;
simulation_TM.input.particles = simulation_TM.input.particles.compute_maximal_particle_distance;

if strcmp(simulation_TM.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(simulation_TM.input.particles.positionArray,simulation_TM.numerics.partitionEdgeSizes);
    simulation_TM.numerics.partitioning = partitioning;
    simulation_TM.numerics.solver.preconditioner.partitioning = partitioning;
    simulation_TM.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    simulation_TM = simulation_TM.numerics.prepareW(simulation_TM);
    simulation_TM.numerics.solver.preconditioner.partitioningIdcs = simulation_TM.numerics.partitioningIdcs;
    simulation_TM.numerics.inverseSolver.preconditioner.partitioningIdcs = simulation_TM.numerics.partitioningIdcs;
end

simulation_TE = simulation_TM;
simulation_TE.input.initialField.polarization = 'TE';

fom = zeros(totalIterations,1);
r_k = parameters(:,1:4);
t_old = zeros(size(r_k));

stored_grad = zeros(6400,totalIterations);
stored_rad = zeros(6400,totalIterations);
for i = 1:totalIterations
    iterationStart = tic;
    [f,g] = CELES_ellipsoid_pol_gradient_iteration(r_k,simulation_TE,points_TE,image_TE,simulation_TM,points_TM,image_TM);
    fom(i) = f;
    gradient = reshape(g,1600,4);
    max_grad = max(gradient);
    max_grad(4) = max_grad(4)*1e2;
    norm_grad = gradient./max_grad;
    t_i = r_k - norm_grad*maxStepSize;
    r_new = t_i+(i-1)/(i+2)*(t_i-t_old);

    axes = r_new(:,1:3);
    axes(axes > maxAxis) = maxAxis;
    axes(axes < minAxis) = minAxis;
    r_new(:,1:3) = axes;
    r_k = r_new;
    
    stored_grad(:,i) = norm_grad(:)*maxStepSize;
    stored_rad(:,i) = r_k(:);
    
    t_old = t_i;
    iterationTime = toc(iterationStart);
    fprintf('\n' + string(iterationTime) + 's for iteration \n');
    
    fom(i)
end

figure
plot(fom)