function shift_param = alignment(img_1,img_2,shift_bits,shift_param)
    now_shift_param=zeros(1,2);
    if shift_bits == 0
        now_shift_param(1) = 0;
        now_shift_param(2) = 0;
    else
        tmp_img_1 = im_shrink(img_1);
        tmp_img_2 = im_shrink(img_2);
        now_shift_param = alignment(tmp_img_1,tmp_img_2,shift_bits-1,now_shift_param);
        now_shift_param(1) = now_shift_param(1)*2;
        now_shift_param(2) = now_shift_param(2)*2;
    end
    
h_1 = size(img_1,1); w_1 = size(img_1,2); h_2 = size(img_2,1); w_2 = size(img_2,2);
tb1 = zeros(h_1, w_1); tb2 = zeros(h_2, w_2); eb1 = zeros(h_1, w_1); eb2 = zeros(h_2, w_2);

[tb1, eb1] = bitmap(img_1);
[tb2, eb2] = bitmap(img_2);

min_err = h_1*w_1;

for i = -1:1
    for j=-1:1
        tmp_x = now_shift_param(1)+i;
        tmp_y = now_shift_param(2)+j;
        shift_tb2 = im_shift(tb2,[tmp_x,tmp_y]);
        shift_eb2 = im_shift(eb2,[tmp_x,tmp_y]);
        diff = xor(tb1,shift_tb2);
        diff = and(diff,eb1);
        diff = and(diff,shift_eb2);
        err = sum(diff(:));
        if err < min_err
            shift_param(1) = tmp_x;
            shift_param(2) = tmp_y;
            min_err = err;
    end
end

end