addpath(genpath('.'))
total_tic = tic;

%initialize global simulation and parts of it
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

totalIterations = 300;
maxStepSize = 50;

lmax = 4;
cuda_compile(lmax);
cuda_compile_T(lmax);

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
refractiveIndex = 3.5;
mediumIndex = 1;

%sphere initial condition
parameters = zeros(900,2);
radiusArray = 400*ones(900,1);
parameters(:,1) = radiusArray;
parameters(:,2) = refractiveIndex;

%upper and lower bounds
maxAxis = 700;
minAxis = 100;

%grid of spheres
xpos = linspace(-22000,22000,30);
ypos = xpos';
zpos = linspace(0,0,1);
[xx,yy,zz] = meshgrid(xpos,ypos,zpos);
positions = zeros(length(radiusArray(:,1)),3);
positions(:,1) = [xx(:)];
positions(:,2) = [yy(:)];
positions(:,3) = [zz(:)];


%particle properties
particles.type = 'sphere';
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
numerics.partitionEdgeSizes = [22200,22200,16000];
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
points = [0,0,100000];
image = [1000];

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
simulation.tables.particleType = simulation.input.particles.type;
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

fom = zeros(totalIterations,1);
r_k = parameters(:,1);
t_old = zeros(size(r_k));

stored_grad = zeros(900,totalIterations);
stored_rad = zeros(900,totalIterations);

for i = 201:300
    iterationStart = tic;
    [f,g] = CELES_gradient_iteration(r_k,simulation,points);
    fom(i) = f;
    norm_gradient_factor = sqrt(sum(g.^2));
    norm_grad = g/norm_gradient_factor;
    final_grad = norm_grad*maxStepSize;
    
    t_i = r_k + final_grad;
    r_new = t_i;
%     r_new = t_i+(i-1)/(i+2)*(t_i-t_old);
    
    r_new(r_new > maxAxis) = maxAxis;
    r_new(r_new < minAxis) = minAxis;
    
    r_k = r_new;
    
    stored_grad(:,i) = final_grad;
    stored_rad(:,i) = r_k(:);
    
    t_old = t_i;
    iterationTime = toc(iterationStart);
    fprintf('\n' + string(iterationTime) + 's for iteration \n');
    fom(i)
end

figure
plot(fom)
    
