addpath(genpath('.'))

simulation = celes_simulation;
simulation2 = celes_simulation;
particles = celes_particles;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner;
inversePreconditioner = celes_preconditioner_new;

lmax = 5;
cuda_compile(lmax);
cuda_compile_T(lmax);

%particle properties
radii = ones(2450,1)*300;
xpos = linspace(-49000/2,49000/2,35);
ypos = xpos';
[xx, yy] = meshgrid(xpos,ypos);
zz = zeros(length(radii),1);
zz(1226:2450) = 2000;
%zz(2451:end) = 4000;
positions = zeros(length(radii),3);
positions(:,1) = [xx(:);xx(:)];
positions(:,2) = [yy(:);yy(:)];
positions(:,3) = zz(:);
refractiveIndex = 1.52;

particles.radiusArray = radii;
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
preconditioner.partitionEdgeSizes = [10000,10000,5200];
inversePreconditioner.type = 'blockdiagonal';
inversePreconditioner.partitionEdgeSizes = [10000,10000,5200];

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
simulation2.tables.pmax = simulation.input.particles.number;

numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
simulation.numerics = numerics;
simulation.tables = celes_tables;
simulation.output = output;
simulation.tables.nmax = simulation.numerics.nmax;
simulation.tables.pmax = simulation.input.particles.number;
simulation2.tables.pmax = simulation.input.particles.number;

y_i = radii;

simulation2 = simulation;
simulation.input.wavelength = 1000;
simulation2.input.wavelength = 775;

fom_coupled = ones(300,1);
fom_uncoupled = ones(300,1);
intensity = ones(300,1);
intensity2 = ones(300,1);

for i = 52:200
    %run both wavelength simulations
    iterationStart = tic;
    simulationStart = tic;
    simulation = simulation.run;
    simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];
    simulation.numerics.solver.preconditioner.masterMatrices = [];
    simulation2 = simulation2.run;
    simulation2.numerics.solver.preconditioner.factorizedMasterMatrices = [];
    simulation2.numerics.solver.preconditioner.masterMatrices = [];
    simulationTime(i) = toc(simulationStart);
    fieldStart = tic;
    %compute figure of merit (not intensity) abs(E_1 + E_2)^2
    E1 = compute_scattered_field_opt(simulation,point);
    E2 = compute_scattered_field_opt(simulation2,point);
    fieldTime = toc(fieldStart);
    %compute intensities for both
    intensity(i) = sum(abs(gather(E1)).^2);
    intensity2(i) = sum(abs(gather(E2)).^2);
    fom_coupled(i) = sum(abs(gather(E1+E2)).^2);
    fom_uncoupled(i) = sum(abs(gather(E1)).^2+abs(gather(E2)).^2);
    %compute derivative of coefficients
    gradCoeffStart = tic;
    simulation = simulation.computeParallelGradMieCoefficients;
    simulation2 = simulation2.computeParallelGradMieCoefficients;   
    gradCoeffTime(i) = toc(gradCoeffStart);
    %compute coupling of scattering coefficients
    simulation = simulation.computeCoupledScatteringCoefficients;
    simulation2 = simulation2.computeCoupledScatteringCoefficients;
    %compute preconditioner for inverse and adjoint fields
    if strcmp(simulation.numerics.inverseSolver.preconditioner.type,'blockdiagonal')
        fprintf(1,'make particle partition...');
        invPartitioning = make_particle_partion(simulation.input.particles.positionArray,simulation.numerics.inverseSolver.preconditioner.partitionEdgeSizes);
        simulation.numerics.inverseSolver.preconditioner.partitioning = invPartitioning;
        fprintf(1,'done\n');
        simulation = simulation.numerics.inverseSolver.preconditioner.prepare(simulation);
    end
    adjointField1Start = tic;
    simulation = simulation.computeChromaticAdjointIntensityFields(E1,E2,point);
    adjointField1Time(i) = toc(adjointField1Start);
    simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
    simulation.numerics.inverseSolver.preconditioner.masterMatrices = [];
    if strcmp(simulation2.numerics.inverseSolver.preconditioner.type,'blockdiagonal')
        fprintf(1,'make particle partition...');
        invPartitioning2 = make_particle_partion(simulation2.input.particles.positionArray,simulation2.numerics.inverseSolver.preconditioner.partitionEdgeSizes);
        simulation2.numerics.inverseSolver.preconditioner.partitioning = invPartitioning;
        fprintf(1,'done\n');
        simulation2 = simulation2.numerics.inverseSolver.preconditioner.prepare(simulation2);
    end
    adjointField2Start = tic;
    simulation2 = simulation2.computeChromaticAdjointIntensityFields(E2,E1,point);
    adjointField2Time(i) = toc(adjointField2Start);
    simulation2.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
    simulation2.numerics.inverseSolver.preconditioner.masterMatrices = [];
    
    %compute individual gradients (uncoupled)
    gradStart = tic;
    grad1Start = tic;
    grad1 = compute_adjoint_grad_intensity(simulation);
    grad1Time(i) = toc(grad1Start);
    grad2Start = tic;
    grad2 = compute_adjoint_grad_intensity(simulation2);
    gradTime(i) = toc(gradStart);
    grad2Time(i) = toc(grad2Start);
    %store them
    %stored_g1(i,:) = grad1;
    %stored_g2(i,:) = grad2;
    %calculate total gradient and store them
    grad = grad1/max(abs(grad1))+grad2/max(abs(grad2));
    %stored_g(i,:) = grad;
    %change radius
    max_step_size = 100;
    %normalize gradient
    full_grad = max_step_size*grad;
    old_rad = simulation.input.particles.radiusArray;
    %stored_rad(i,:) = old_rad;
    %new_rad = old_rad + full_grad;
    r_0 = old_rad;
    radius = y_i+full_grad;
    radius(radius < 150) = 151;
    radius(radius > 700) = 699;
    y_i = radius+(i-1)/(i+2).*(radius-r_0);
    radius = y_i;
    
    simulation.input.particles.radiusArray = radius;
    simulation2.input.particles.radiusArray = radius;
    iterationTime(i) = toc(iterationStart);
end

figure
plot(fom_coupled);
title('fom');
hold on
plot(fom_uncoupled);

figure
plot(intensity)
hold on
plot(intensity2)
hold on
plot(fom_uncoupled)



[x,z] = meshgrid(-50000:200:50000,-1000:200:110000); y=x-x;
output.fieldPoints = [x(:),y(:),z(:)];
fieldPoints = [x(:),y(:),z(:)];


output.fieldPointsArrayDims = size(x);
simulation.output = output;
simulation=simulation.evaluateFields;
figure
plot_field(gca,simulation,'abs E','Scattered field',simulation.input.particles.radiusArray)
colormap(jet)
caxis([0 5])
colorbar
output.fieldPointsArrayDims = size(x);
simulation2.output = output;
simulation2=simulation2.evaluateFields;
figure
plot_field(gca,simulation2,'abs E','Scattered field',simulation2.input.particles.radiusArray)
colormap(jet)
caxis([0 5])
colorbar
figure
plot_spheres(gca,simulation.input.particles.positionArray,simulation.input.particles.radiusArray,simulation.input.particles.refractiveIndexArray,'view xy')
%
