param_o = [500,300,400];
param_a = [501,300,400];
param_b = [500,301,400];
param_c = [500,300,401];

lmax = 4;
dphi = 40;
dtheta = 40;
ni = 1;
ns = 1.52;
wavelength = 800;

[T,dT] = compute_T(lmax,dtheta,dphi,param_o,ni,ns,wavelength);
[Ta,dTa] = compute_T(lmax,dtheta,dphi,param_a,ni,ns,wavelength);
[Tb,dbT] = compute_T(lmax,dtheta,dphi,param_b,ni,ns,wavelength);
[Tc,dTc] = compute_T(lmax,dtheta,dphi,param_c,ni,ns,wavelength);

delTa = T-Ta;
delTb = T-Tb;
delTc = T-Tc;

figure
subplot(3,1,1)
imagesc(abs(delTa))
colorbar
subplot(3,1,2)
imagesc(abs(delTb))
colorbar
subplot(3,1,3)
imagesc(abs(delTc))
colorbar


figure
subplot(3,1,1)
imagesc(abs(dT(:,:,1)+delTa))
colorbar
subplot(3,1,2)
imagesc(abs(dT(:,:,2)+delTb))
colorbar
subplot(3,1,3)
imagesc(abs(dT(:,:,3)+delTc))
colorbar
