addpath(genpath('.'))

simulation = celes_simulation;
particles = celes_particles2;
input = celes_input;
tables = celes_tables;
initialField = celes_initialField;
numerics = celes_numerics;

particles.type = 'ellipsoid';

lmax = 4;
numParts = 2;

axis_a = 1100*ones(numParts,1);
axis_b = 600*ones(numParts,1);
axis_c = 900*ones(numParts,1);
angles = linspace(0,pi/500,numParts);
position = rand(numParts,3);
refractiveIndex = ones(numParts,1)*1.52;
parameters = zeros(numParts,5);
parameters(:,1) = axis_a;
parameters(:,2) = axis_b;
parameters(:,3) = axis_c;
parameters(:,4) = angles;
parameters(:,5) = refractiveIndex;

particles.positionArray = position;
particles.parameterArray = parameters;

% polar angle of incoming beam/wave, in radians (for Gaussian beams, 
% only 0 and pi are currently possible)
initialField.polarAngle = 0;

% azimuthal angle of incoming beam/wave, in radians
initialField.azimuthalAngle = 0;

% polarization of incoming beam/wave ('TE' or 'TM')
initialField.polarization = 'TM';

% width of beam waist (use 0 or inf for plane wave)
initialField.beamWidth = 0;

% vacuum wavelength (same unit as particle positions and radius)
input.wavelength = 1500;
input.mediumRefractiveIndex = 1;

numerics.lmax = lmax;

input.initialField = initialField;
simulation.numerics = numerics;
simulation.tables = celes_tables;
input.particles = particles;
simulation.input = input;
simulation.tables.pmax = simulation.input.particles.number;
simulation.tables.nmax = simulation.numerics.nmax;

nmax = simulation.tables.nmax;
singleMie = tic;
simulation = simulation.computeParallelMieCoefficients;
timeSingle = toc(singleMie);
simulation = simulation.computeGradMieCoefficients;
% parMie = tic;
% simulation = simulation.computeParallelMieCoefficients;
% timePar = toc(parMie);

mieCoeff = simulation.tables.mieCoefficients;
gradMieCoeff = simulation.tables.gradMieCoefficients;
% figure
% subplot(2,1,1)
% imagesc(radius,1:100,real(mieCoeff))
% title('real mieCoeff');
% colorbar
% subplot(2,1,2)
% imagesc(radius,1:100,imag(mieCoeff))
% title('imag mieCoeff');
% colorbar

% figure
% subplot(2,1,1)
% imagesc(1:nmax,radius,abs(mieCoeff));
% title('abs mieCoeff');
% colorbar
% subplot(2,1,2)
% imagesc(1:nmax,radius,angle(mieCoeff));
% title('phase mieCoeff');
% colorbar
% 
% figure
% imagesc(1:nmax,radius,abs(mieCoeff));
% title('abs mieCoeff');
% colormap('gray')
% colorbar

diff_angles = diff(angles);
angular_diff = diff_angles(1);
angular_derivs = gradMieCoeff(:,:,:,4);

T0 = squeeze(mieCoeff(1,:,:));
T1 = squeeze(mieCoeff(2,:,:));
T01 = T0 + squeeze(angular_derivs(1,:,:))*angular_diff;
dT01 = T01-T1;

figure
subplot(2,2,1)
imagesc(abs(T1))
colorbar
title('T1');
axis square
subplot(2,2,2)
imagesc(abs(T01))
colorbar
title('T01')
axis square
subplot(2,2,3)
imagesc(abs(squeeze(angular_derivs(1,:,:))))
colorbar
title('dT0');
axis square
subplot(2,2,4)
imagesc(abs(dT01))
colorbar
title('dT01');
axis square

figure
imagesc(100*abs(dT01)./abs(T1))
colorbar

figure
subplot(2,1,1)
imagesc(abs(T1-T0))
colorbar
title('T1-T0');
axis square
subplot(2,1,2)
imagesc(abs(squeeze(angular_derivs(1,:,:))*angular_diff))
colorbar
title('dT0')
axis square

norm_ang = squeeze(angular_derivs(1,:,:))/max(max(abs(squeeze(angular_derivs(1,:,:)))));
norm_diff_ang = (T1-T0)/max(max(abs(T1-T0)));

figure
subplot(3,1,1)
imagesc(abs(norm_ang))
title('analytic derivative');
colorbar
axis square
subplot(3,1,2)
imagesc(abs(norm_diff_ang))
title('numerical derivative');
colorbar
axis square
subplot(3,1,3)
imagesc(abs(norm_ang-norm_diff_ang))
title('difference');
colorbar
axis square