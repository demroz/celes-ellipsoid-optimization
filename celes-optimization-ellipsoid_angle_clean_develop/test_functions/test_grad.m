lmax = 4;
ellip = [700,700,1000];
ellip_max = 1100;
ellip_min = 150;
wavelength = 1550;
ni = 1;
ns = 1.52;

grad = zeros(1,3);

a_in = a_input.';
ellip_hist = ones(200,3);
b_hist = ones(200,48);


for i = 1:50
    [T_i,dT_i] = compute_T(lmax,40,40,ellip,ni,ns,wavelength);
    b_iout = T_i*a_in;
    b_hist(i,:) = b_iout;
    fom(i) = sum(abs(b_iout).^2);
    db_ia = squeeze(dT_i(:,:,1))*a_in;
    db_ib = squeeze(dT_i(:,:,2))*a_in;
    db_ic = squeeze(dT_i(:,:,3))*a_in;
    dfom_db = conj(b_iout)*sum(b_iout);
    grad(1) = 2*real(sum(dfom_db.*db_ia));
    grad(2) = 2*real(sum(dfom_db.*db_ib));
    grad(3) = 2*real(sum(dfom_db.*db_ic));
    norm_grad = grad/sqrt(sum(grad.^2));
    ellip = ellip-norm_grad*20;
    ellip(ellip > ellip_max) = ellip_max-5;
    ellip(ellip < ellip_min) = ellip_min+5;
    ellip_hist(i,:) = ellip;
end

figure
plot(fom);

figure
imagesc(abs(b_hist))
colorbar
figure
imagesc(ellip_hist)
colorbar