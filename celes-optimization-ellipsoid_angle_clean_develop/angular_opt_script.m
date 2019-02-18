%add entire folder to search path
addpath(genpath('.'))


%>-------------------------------------------------------------------------
%                   Begin user editable properties
%>-------------------------------------------------------------------------


%maximum expansion order determined by particle size, refractiv index and
%wavelength of interest
lmax = 4;
totalGradientSteps = 100;
maxStepSize = 100;

%figure of merit
points = [0,0,10000];

%excitation beam properties
wavelength = 1550;
polarization = 'TE';
%beamwidth of faux gaussian beam, 0 or inf for plane wave
beamWidth = 0;
%focus of a gaussian beam
focalPoint = [0,0,0];
%tilt
polarAngle = 0;
azimuthalAngle = 0;

%refractive indices
mediumIndex = 1;
particleIndex = 1.52;

%particle properties
maxAxis = 1200;
minAxis = 500;
initialAxis = 700;
particleType = 'sphere';
numberParticlesLinear = 30;

%numerical settings
angleDiscretization = 1e3;
gpuFlag = true;
particleDistanceResolution = 0.5;

%solver settings
solverType = 'BiCGSTAB';
maxIterations = 1000;
solverTolerance = 1e-3;
%preconditioner settings
preconditionerType = 'blockdiagonal';
partitionEdgeSizes = [30000,30000,30000];
%only for GMRES type solver
GMRES_restart = 10;


%>-------------------------------------------------------------------------
%                       End user editable properties
%>-------------------------------------------------------------------------


%setup all simulation objects
simulation = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
tables = celes_tables;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

cuda_compile(lmax);
cuda_compile_T(lmax);

%set position grid of the particles
%automatically create square grid of particles with 
%periodicity >~ maxRadius*2
xpos = (-maxAxis*numberParticlesLinear+maxAxis/2):(maxAxis+25)*2:(maxAxis*numberParticlesLinear-maxAxis/2);
ypos = xpos';
zpos = 0;

[xx,yy,zz] = meshgrid(xpos,ypos,zpos);
positions = [xx(:),yy(:),zz(:)];

%set parameters for particles
numParticles = numberParticlesLinear^2;
parameters = zeros(numParticles,2);
parameters(:,1) = initialAxis*ones(numParticles,1);
parameters(:,2) = particleIndex*ones(numParticles,1);

particles.type = particleType;
particles.parameterArray = parameters;
particles.positionArray = positions;

%set parameters for initial field
initialField.polarAngle = polarAngle;
initialField.azimuthalAngle = azimuthalAngle;
initialField.polarization = polarization;
initialField.beamWidth = beamWidth;
initialField.focalPoint = focalPoint;

%assemble input object
input.wavelength = wavelength;
input.mediumRefractiveIndex = mediumIndex;
input.particles = particles;
input.initialField = initialField;

%set parameters for preconditioner
preconditioner.type = preconditionerType;
inversePreconditioner.type = preconditionerType;

%set solver properties
solver.type = solverType;
solver.tolerance = solverTolerance;
solver.maxIter = maxIterations;
solver.restart = GMRES_restart;
inverseSolver = solver;
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%assemble numerics object and set numeric properties
numerics.lmax = lmax;
numerics.particleDistanceResolution = particleDistanceResolution;
numerics.gpuFlag = gpuFlag;
numerics.polarAnglesArray = 0:pi/angleDiscretization:pi;
numerics.azimuthalAnglesArray = 0:pi/angleDiscretization:2*pi;
numerics.partitionEdgeSizes = partitionEdgeSizes;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;

%assemble some table data
tables.nmax = numerics.nmax;
tables.pmax = input.particles.number;

%assemble simulation object
simulation.input = input;
simulation.numerics = numerics;
simulation.tables = tables;
simulation.output = output;
simulation = simulation.computeInitialFieldPower;
simulation = simulation.computeTranslationTable;
simulation.input.particles = simulation.input.particles.compute_maximal_particle_distance;

%precompute W matrices for preconditioner and store
if strcmp(simulation.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(simulation.input.particles.positionArray,simulation.numerics.partitionEdgeSizes);
    simulation.numerics.partitioning = partitioning;
    fprintf(1,' done\n');
    simulation = simulation.numerics.prepareW(simulation);
end

fom = zeros(totalGradientSteps,1);

%begin optimization process loop
for i = 1:totalGradientSteps
    
    %compute mie coefficients
    simulation = simulation.computeParallelMieCoefficients;
    
    %prepare preconditioner
    simulation = simulation.numerics.solver.preconditioner.prepareM(simulation);
    
    %find initial field coefficients corresponding to input beam
    simulation = simulation.computeInitialFieldCoefficients;
    
    %solve for scattered field coefficients
    simulation = simulation.computeScatteredFieldCoefficients;
    
    %clear M preconditioner from memory
    simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];
    
    %compute figure of merit
    E = compute_scattered_field_opt(simulation,points);
    fom(i) = sum(sum(abs(E).^2,2));
    
    %compute grad mie coefficients, and coupled scattered field
    %coefficients
    simulation = simulation.computeParallelGradMieCoefficients;
    simulation = simulation.computeCoupledScatteringCoefficients;
    
    %prepare inverse preconditioner
    simulation = simulation.numerics.inverseSolver.preconditioner.prepareMt(simulation);
    
    %compute adjoint coefficients
    simulation = simulation.computeAdjointFields(E,points);
    
    %clear Mt preconditioner from memeory
    simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrcies = [];
    
    %compute the normalized gradient
    grad = compute_adjoint_grad_intensity(simulation);
    norm_grad = grad/sum(abs(grad).^2);
    final_grad = norm_grad*maxStepSize;
    
    %update the stored radii
    r_0 = simulation.input.particles.parameterArray(:,1);
    r_new = r_0 + final_grad;
    simulation.input.particles.parameterArray(:,1) = r_new;
end