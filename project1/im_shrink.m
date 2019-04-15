function    shrink_img = im_shrink(img)
    h = size(img,1);
    w = size(img,2);
    tmp_h = int32(h/2);
    tmp_w = int32(w/2);
    if mod(h,2) ~= 0
        tmp_h = tmp_h-1;
    end
    if mod(w,2) ~= 0
        tmp_w = tmp_w-1;
    end
    shrink_img = zeros(tmp_h,tmp_w);
    
    for i=1:tmp_h
        for j=1:tmp_w
            shrink_img(i,j) = img(i*2,j*2);
        end
    end
end