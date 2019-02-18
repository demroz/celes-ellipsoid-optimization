load('LBFGS_helix_3um_final.mat');

radFile = fopen('LBFGS_large_helix_3um_rad.txt','w');
posFile = fopen('LBFGS_large_helix_3um_offpos.txt','w');

positions(:,3) = r_f;

fprintf(radFile,'%4.0f\n',r_f);
fclose(radFile);
fprintf(posFile,'%4.0f, %4.0f, %4.0f\n',positions.');
fclose(posFile);