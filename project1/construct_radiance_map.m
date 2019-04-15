function radiance_map = construct_radiance_map(images, g, ln_t, w)
    [row, col, channel, number] = size(images);
    E = zeros(row,col,channel);
    
    for c = 1:3
        for i = 1:row
            for j = 1:col
                sum_1=0;
                sum_2=0;
                for k = 1:number
                    z = images(i,j,c,k)+1;
                    sum_1 = sum_1 + w(z)*(g(z)-ln_t(k));
                    sum_2 = sum_2 + w(z);
                end
                E(i,j,c) = sum_1 / sum_2;
            end
        end
    end
    
    E(isnan(E))=0;
    radiance_map = exp(E);
    index = find(isnan(radiance_map)|isinf(radiance_map));
    radiance_map(index) = 0;
end
