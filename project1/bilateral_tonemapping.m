function ldr_img = bilateral_tonemapping(radiance_map)
    h = size(radiance_map,1);
    w = size(radiance_map,2);
    intensity = zeros(h,w);
    log_intensity = zeros(h,w);
    color = zeros(h,w,3);
    log_base = zeros(h,w);
    log_detail = zeros(h,w);
    ldr_img = zeros(h,w,3);
    
    color_sigma = 0.4;
    if h>w
        spatial_sigma = 0.02*h;
    else
        spatial_sigma = 0.02*w;
    end
    
    for i=1:h
        for j=1:w
            intensity(i,j) = 0.212671*radiance_map(i,j,1) + 0.715160*radiance_map(i,j,2) + 0.072169*radiance_map(i,j,3);
            log_intensity(i,j) = log(intensity(i,j));
        end
    end
    
    for i=1:h
        for j=1:w
            for c=1:3
                color(i,j,c) = radiance_map(i,j,c)/intensity(i,j);
            end
        end
    end
    
    log_base = b_filter(log_intensity,color_sigma,spatial_sigma);
    
    for i=1:h
        for j=1:w
            log_detail(i,j) = log_intensity(i,j)-log_base(i,j);
        end
    end
    
    max_log_base = log_base(1,1);
    min_log_base = log_base(1,1);
    
    for i=1:h
        for j=1:w
            if max_log_base<log_base(i,j)
                max_log_base = log_base(i,j);
            end
            if min_log_base>log_base(i,j)
                min_log_base = log_base(i,j);
            end
        end
    end
    
    disp(max_log_base);
    disp(min_log_base);
    
    targetcontrast = 5.0;
    compressionfactor  = log10(targetcontrast) / (max_log_base-min_log_base);
    absolutescale = max_log_base * compressionfactor;
    
    contrast_reduce = zeros(h,w);
    for i=1:h
        for j=1:w
            contrast_reduce(i,j) = log_base(i,j) * compressionfactor + log_detail(i,j) - absolutescale;
            contrast_reduce(i,j) = power(10,contrast_reduce(i,j));
            for c=1:3
                ldr_img(i,j,c) = power(contrast_reduce(i,j) * color(i,j,c),1.0/1.6);
            end
        end
    end
end