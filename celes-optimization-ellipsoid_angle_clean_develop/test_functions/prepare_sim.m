addpath(genpath('.'))

simulation = celes_simulation;
particles = celes_particles;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_new;

lmax = 3;
% cuda_compile(lmax);
% cuda_compile_T(lmax);

%particle properties
partX = 6;
partY = 6;
numParticles = partX*partY;
% a = [100,100,100,100];
% b = [100,200,100,200];
% c = [300,300,300,100];
a = rand(36,1)*400;
b = a;
c = a;
radii = a;
xpos = linspace(-partX*600,partX*600,partX);
ypos = linspace(-partY*600,partY*600,partY);
[xx, yy] = meshgrid(xpos,ypos);
zz = zeros(length(radii),1);
positions = zeros(length(radii),3);
positions(:,1) = xx(:);
positions(:,2) = yy(:);
positions(:,3) = zz(:);
refractiveIndex = 1.52;

particles.radiusArray = radii;
%particles.type = 'ellipsoid';
particles.type = 'sphere';
particles.refractiveIndexArray = refractiveIndex*ones(1,length(radii));
particles.positionArray = positions;

%numeric properties
numerics.lmax = lmax;
numerics.particleDistanceResolution = 1;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

point = [0,0,100000];

%solver properties
solver.type = 'BiCGStab';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;

%preconditioner properties
preconditioner.type = 'blockdiagonal';
preconditioner.partitionEdgeSizes = [30000,30000,5200];
inversePreconditioner.type = 'blockdiagonal';
inversePreconditioner.partitionEdgeSizes = [30000,30000,5200];

%inptu into solver
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;
initialField.polarization = 'TE';
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%input properties minus wavelength
input.mediumRefractiveIndex = 1;

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

y_i = radii;

simulation.input.wavelength = 1000;
tic;
simulation = simulation.computeParallelMieCoefficients;
toc;
simulation = simulation.computeTranslationTable;

fprintf(1,'compute maximal particle distance ...');
simulation.input.particles = simulation.input.particles.compute_maximal_particle_distance;
fprintf(1,' done\n');
tprec=tic;
if strcmp(simulation.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(simulation.input.particles.positionArray,simulation.numerics.solver.preconditioner.partitionEdgeSizes);
    simulation.numerics.solver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    simulation = simulation.numerics.solver.preconditioner.prepareW(simulation);
end
simulation.output.preconiditionerPreparationTime = toc(tprec);
fprintf(1,'preconditioner prepared in %.1f seconds.\n',simulation.output.preconiditionerPreparationTime);
tsolv=tic;

allM = simulation.numerics.solver.preconditioner.returnM(simulation);

figure
imagesc(abs(allM{1}));
colorbar

M = prepareM(simulation);
figure
imagesc(abs(M));
colorbar

figure
imagesc(abs(allM{1}-M));
colorbar
