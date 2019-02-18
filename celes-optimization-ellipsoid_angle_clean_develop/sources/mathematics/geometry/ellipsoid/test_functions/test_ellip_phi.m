clear all;
close all;

[J110,J120,J210,J220,dJ110,dJ120,dJ210,dJ220] = compute_J_ellip_celes_posm_2(4,30,30,500,600,800,0,1,1.52,900,3);
[J111,J121,J211,J221,dJ111,dJ121,dJ211,dJ221] = compute_J_ellip_celes_posm_2(4,30,30,500,600,800,0.01,1,1.52,900,3);
dJ11 = J110-J111;
dJ12 = J120-J121;
dJ21 = J210-J211;
dJ22 = J220-J221;

figure
subplot(2,1,1)
imagesc(abs(dJ11)/max(max(abs(dJ11))))
colorbar
subplot(2,1,2)
imagesc(abs(dJ110(:,:,4))/max(max(abs(dJ110(:,:,4)))))
colorbar
figure
subplot(2,1,1)
imagesc(abs(dJ12)/max(max(abs(dJ12))))
colorbar
subplot(2,1,2)
imagesc(abs(dJ120(:,:,4))/max(max(abs(dJ120(:,:,4)))))
colorbar
figure
subplot(2,1,1)
imagesc(abs(dJ21)/max(max(abs(dJ21))))
colorbar
subplot(2,1,2)
imagesc(abs(dJ210(:,:,4))/max(max(abs(dJ210(:,:,4)))))
colorbar
figure
subplot(2,1,1)
imagesc(abs(dJ22)/max(max(abs(dJ22))))
colorbar
subplot(2,1,2)
imagesc(abs(dJ220(:,:,4))/max(max(abs(dJ220(:,:,4)))))
colorbar

[T0 dT0] = compute_T_2(4,30,30,[500,600,700,0],1,1.52,900);
[T1 dT1] = compute_T_2(4,30,30,[500,600,700,0.1],1,1.52,900);

% dT01 = T1-T0;
% 
% figure
% subplot(2,1,1)
% imagesc(abs(dT01));
% colorbar
% subplot(2,1,2)
% imagesc(abs(dT0(:,:,4)));
% colorbar

T1p = T0+dT0(:,:,4)*0.1;
figure
subplot(3,1,1)
imagesc(imag(T1p))
colorbar
subplot(3,1,2)
imagesc(imag(T1))
colorbar
subplot(3,1,3)
imagesc(imag(T1p-T1));
colorbar

figure
subplot(3,1,1)
imagesc(real(T1p))
colorbar
subplot(3,1,2)
imagesc(real(T1))
colorbar
subplot(3,1,3)
imagesc(real(T1p-T1));
colorbar


[T0 dT0] = compute_T_2(4,30,30,[500,600,700,0],1,1.52,900);
[T1 dT1] = compute_T_2(4,30,30,[500,621,700,0],1,1.52,900);

dT01 = T1-T0;
T1p = T0+21*dT0(:,:,2);

figure
subplot(3,1,1)
imagesc(imag(T1p))
colorbar
subplot(3,1,2)
imagesc(imag(T1))
colorbar
subplot(3,1,3)
imagesc(imag(T1p-T1));
colorbar

figure
subplot(3,1,1)
imagesc(real(T1p))
colorbar
subplot(3,1,2)
imagesc(real(T1))
colorbar
subplot(3,1,3)
imagesc(real(T1p-T1));
colorbar