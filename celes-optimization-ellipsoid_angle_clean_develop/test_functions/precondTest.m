addpath(genpath('.'))

simulation = celes_simulation;
simulation_v1 = celes_simulation;
particles = celes_particles;
initialField = celes_initialField;
input = celes_input;
input_v1 = celes_input;
numerics = celes_numerics_v2;
numerics_v1 = celes_numerics;
solver = celes_solver;
solver_v1 = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner_v1 = celes_preconditioner;
preconditioner = celes_preconditioner_v2;
inversePreconditioner_v1 = celes_preconditioner;
inversePreconditioner = celes_preconditioner_v2;

lmax = 3;
cuda_compile(lmax);
cuda_compile_T(lmax);

%particle properties
radii = ones(100,1)*500;
radiusArray = [radii,radii,radii];
max_rad = 1600;
min_rad = 400;
xpos = linspace(-6000,6000,10);
ypos = xpos';
[xx, yy] = meshgrid(xpos,ypos);
zz = zeros(length(radii),1);
positions = zeros(length(radii),3);
positions(:,1) = xx(:);
positions(:,2) = yy(:);
positions(:,3) = zz(:);
refractiveIndex = 1.52;

particles.type = 'ellipsoid';
particles.radiusArray = radiusArray;
particles.refractiveIndexArray = refractiveIndex*ones(1,length(radii));
particles.positionArray = positions;

input.particles = particles;

particles.type = 'sphere';
particles.radiusArray = radii;
input_v1.particles = particles;

%numeric properties
numerics.lmax = lmax;
numerics.particleDistanceResolution = 1;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;
numerics_v1.lmax = lmax;
numerics_v1.particleDistanceResolution = 1;
numerics_v1.gpuFlag = true;
numerics_v1.polarAnglesArray = 0:pi/1e3:pi;
numerics_v1.azimuthalAnglesArray = 0:pi/1e3:2*pi;


point = [0,0,100000];

%solver properties
solver.type = 'BiCGStab';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;
solver_v1.type = 'BiCGStab';
solver_v1.tolerance = 1e-3;
solver_v1.maxIter = 1000;
solver_v1.restart = 1000;
inverseSolver_v1 = solver_v1;

%preconditioner properties
preconditioner.type = 'blockdiagonal';
numerics.partitionEdgeSizes = [10000,10000,6200];
inversePreconditioner.type = 'blockdiagonal';
preconditioner_v1.type = 'blockdiagonal';
preconditioner_v1.partitionEdgeSizes = [10000,10000,6200];
inversePreconditioner_v1.type = 'blockdiagonal';


%inptu into solver
solver.preconditioner = preconditioner;
solver_v1.preconditioner = preconditioner_v1;
inverseSolver.preconditioner = inversePreconditioner;
inverseSolver_v1.preconditioner = inversePreconditioner_v1;

%initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
initialField.polarization = 'TE';
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%input properties minus wavelength
input.mediumRefractiveIndex = 1;
input_v1.mediumRefractiveIndex = 1;

%put into simulation object;
input.initialField = initialField;
input_v1.initialField = initialField;
simulation.input = input;
simulation_v1.input = input_v1;
simulation.tables.pmax = simulation.input.particles.number;
simulation_v1.tables.pmax = simulation_v1.input.particles.number;

numerics.solver = solver;
numerics_v1.solver = solver_v1;
numerics.inverseSolver = inverseSolver;
numerics_v1.inverseSolver = inverseSolver_v1;
simulation.numerics = numerics;
simulation_v1.numerics = numerics_v1;
simulation.tables = celes_tables;
simulation_v1.tables = celes_tables;
simulation.output = output;
simulation_v1.output = output;
simulation.tables.nmax = simulation.numerics.nmax;
simulation.tables.pmax = simulation.input.particles.number;
simulation_v1.tables.nmax = simulation_v1.numerics.nmax;
simulation_v1.tables.pmax = simulation_v1.input.particles.number;

y_i = radii;

simulation.input.wavelength = 1550;
simulation_v1.input.wavelength = 1550;

fom = zeros(300,1);

simulation = simulation.computeInitialFieldPower;
simulation = simulation.computeTranslationTable;
simulation.input.particles = simulation.input.particles.compute_maximal_particle_distance;
simulation_v1 = simulation_v1.computeInitialFieldPower;
simulation_v1 = simulation_v1.computeTranslationTable;
simulation_v1.input.particles = simulation_v1.input.particles.compute_maximal_particle_distance;

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
simulation_v1.numerics.solver.preconditioner.partitioning = partitioning;
simulation = simulation.computeParallelMieCoefficients;
simulation_v1 = simulation_v1.computeParallelMieCoefficients;
%Compute scattered field coefficients b_i,n for lambda 1
precTime = tic;
simulation = simulation.numerics.solver.preconditioner.prepareM(simulation);
simulation_v1 = simulation_v1.numerics.solver.preconditioner.prepare(simulation_v1);
M = simulation.numerics.solver.preconditioner.returnM(simulation);
fprintf(1,'preconditioner prepared in %.1f seconds.\n',toc(precTime));
solvTime = tic;
simulation = simulation.computeInitialFieldCoefficients;
simulation_v1 = simulation_v1.computeInitialFieldCoefficients;