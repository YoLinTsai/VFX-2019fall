function ldr = r_tonemapping(r_map) %global only

    alpha = 0.3;
    delta = 1e-6;
    white = 100;
    h = size(r_map,1);
    w = size(r_map,2);
    ldr = zeros(h,w,3);
    Lm = zeros(h,w);
    Ld = zeros(h,w);
    Cw = zeros(h,w,3);

    intensity = 0.21*r_map(:,:,1) + 0.71*r_map(:,:,2) + 0.08*r_map(:,:,3);
    mean_intensity = exp(mean(mean(log(delta+intensity))));
    Lm = (alpha/mean_intensity)*intensity;
    for i=1:h
        for j=1:w
            for c=1:3
                Ld(i,j) = (Lm(i,j)*(1+Lm(i,j)/(white*white)))/(1+Lm(i,j));
            end
        end
    end
    
    for i=1:h
        for j=1:w
            for c=1:3
                Cw(i,j,c) = r_map(i,j,c)/intensity(i,j);
                ldr(i,j,c) = Cw(i,j,c)*double(Ld(i,j));
            end
        end
    end
end