%add entire folder to search path
addpath(genpath('.'))


%>-------------------------------------------------------------------------
%                   Begin user editable properties
%>-------------------------------------------------------------------------
load('test_angle.mat')

%maximum expansion order determined by particle size, refractiv index and
%wavelength of interest
lmax = 4;
totalGradientSteps = 100;
maxStepSize = 1;

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

initialOrientation = 90;
particleType = 'ellipsoid';

%numerical settings
angleDiscretization = 1e3;
gpuFlag = true;
particleDistanceResolution = 0.5;

%solver settings
solverType = 'BiCGStab';
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

parameters(:,4) = initialOrientation;
parameters(:,5) = particleIndex;

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

t_old = zeros(size(parameters(:,4)));

%precompute W matrices for preconditioner and store
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

fom = zeros(totalGradientSteps,1);

%begin optimization process loop
for i = 1:(totalGradientSteps)
    
    %compute mie coefficients
    simulation = simulation.computeParallelMieCoefficients;
    
    %prepare preconditioner
    simulation = simulation.numerics.solver.preconditioner.prepareM(simulation);
    
    %find initial field coefficients corresponding to input beam
    simulation = simulation.computeInitialFieldCoefficients;
    
    %solve for scattered field coefficients
    simulation = simulation.computeScatteredFieldCoefficients();
    
    %clear M preconditioner from memory
    simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];
    
    %compute figure of merit
    E = compute_scattered_field_opt(simulation,field_pts);
    fom(i) = sum((absE2-sum(abs(gather(E)).^2,2)).^2);
    
    %compute grad mie coefficients, and coupled scattered field
    %coefficients
    simulation = simulation.computeParallelGradMieCoefficients;
    simulation = simulation.computeCoupledScatteringCoefficients;
    
    %prepare inverse preconditioner
    simulation = simulation.numerics.inverseSolver.preconditioner.prepareMt(simulation);
    
    %compute adjoint coefficients
    simulation = simulation.computeImageAdjointCoefficients(absE2,E,field_pts);
    
    %clear Mt preconditioner from memeory
    simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
    
    %compute the normalized gradient
%     %axial gradient
%     grad(:,1) = compute_adjoint_grad_intensity(simulation,1);
%     grad(:,2) = compute_adjoint_grad_intensity(simulation,2);
%     grad(:,3) = compute_adjoint_grad_intensity(simulation,3);
    %angular gradient
    grad = compute_adjoint_grad_intensity(simulation,4);
    
    max_grad = max(grad)*1e2;
    
    norm_grad = grad./max_grad;
    final_grad = maxStepSize*norm_grad;
    a_i = simulation.input.particles.parameterArray(:,4);
    %update the stored radii
    t_i = a_i - final_grad;
    a_new = t_i+(i-1)/(i+2)*(t_i-t_old);
%     r_new = r_i + final_grad;
%     
    
    stored_rad(:,i) = a_i(:);
    stored_grad(:,i) = final_grad(:);
    
    simulation.input.particles.parameterArray(:,4) = a_new;
    t_old = t_i;
end

