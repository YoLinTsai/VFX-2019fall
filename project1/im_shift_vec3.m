function shift_img = im_shift_vec3(img,shift_param)
    h = size(img,1);
    w = size(img,2);
    tmp_x = int32(shift_param(1));
    tmp_y = int32(shift_param(2));
    shift_img = zeros(h,w,3);
    for i=1:h
        for j=1:w
            for c=1:3
                if i+tmp_x>0 && i+tmp_x<=h && j+tmp_y>0 && j+tmp_y<=w
                    shift_img(i,j,c) = img(i+tmp_x,j+tmp_y,c);
                end
            end
        end
    end
end