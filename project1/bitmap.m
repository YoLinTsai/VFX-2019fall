function [tb, eb] = bitmap(img)
    h = size(img,1) ;
    w = size(img,2);
    histogram = zeros(1,256);
    median_counter = h*w/2;
    tb = zeros(h, w); 
    eb = zeros(h, w);
    
    for i=1:h
        for j=1:w
            histogram(int32(img(i,j)+1)) = histogram(int32(img(i,j)+1)) +1;
        end
    end
    
    for i=1:256
        if median_counter - histogram(i) > 0
            median_counter = median_counter - histogram(i);
        else
            threshold = i; break;
        end
    end
    
    for i=1:h
        for j=1:w
            if img(i,j) <= threshold
                tb(i,j) = 0;
            else
                tb(i,j) = 1;
            end
            if img(i,j)<=threshold+4 && img(i,j)>=threshold-4
                eb(i,j) = 0;
            else
                eb(i,j) =1;
            end
        end
    end
    
end