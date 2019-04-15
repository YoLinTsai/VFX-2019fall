function log_base = b_filter(log_intensity,color_sigma,spatial_sigma)

    h = size(log_intensity,1);
    w = size(log_intensity,2);
    log_base = zeros(h,w);
    twice_cs = 2*color_sigma*color_sigma;
    twice_ss = 2*spatial_sigma*spatial_sigma;
    three_ss = 3*spatial_sigma;
    
    for i=1:h
        for j=1:w
            wp = 0;
            
            qi_min = i-three_ss;
            qi_max = i+three_ss;
            qj_min = j-three_ss;
            qj_max = j+three_ss;
            
            qi_min = int32(qi_min);
            qj_min = int32(qj_min);
            qi_max = int32(qi_max);
            qj_max = int32(qj_max);
            
            for ni=qi_min:qi_max
                if ni<1
                    continue; 
                end
                if ni>h
                    break; 
                end
                for nj=qj_min:qj_max
                    if nj<1
                        continue;
                    end
                    if nj>w
                        break; 
                    end
                    space2 = (i-ni)*(i-ni) + (j-nj)*(j-nj);
                    range = log_intensity(i,j) - log_intensity(ni,nj);
                    n_w = exp(double(-space2/twice_ss))*exp(double(-(range*range)/twice_cs));
                    log_base(i,j) = log_base(i,j)+n_w*log_intensity(ni,nj);
                    wp = wp+n_w;
                end
            end
        end
        if mod(i,100) == 0
            disp('now');
            disp(i);
        end
    end
    log_base=log_base/wp;
end