function y= calcdesc(vecs)

vecs=vecs./sum(vecs);

L1=vecs(1);
L2=vecs(2);
L3=vecs(3);
% if vecs(3)<=0 || vecs(2)<=0 || vecs(1)<=0
% %     disp('warning: eigenvalue equal to or less than zero')
%     
% end

omni = (L1*L2*L3)^(1/3);
eigent = -(L1*log(L1)+L2*log(L2)+L3*log(L3));
aniso = (L1-L3)/L1;
planar = (L2-L3)/L1;
linear = (L1-L2)/L1;
curv = L3;
scatt = L3/L1;

y = [omni,eigent,aniso,planar,linear,curv,scatt,vecs'];

% if sum(imag(y)>0)
% %     disp(['warning: imaginary value calculated for point ', num2str(i)])
% end

