function radiance_map = myhdr(img_data_filename,image_num_filename,point_num,shift_bits)
    
    if point_num <27
        disp('At least choose 27 points for g fuction solving.');
        point_num = 27;
    end
    lambda = 10;

    [choosing_point_image,g_images,images,e]=readImages(img_data_filename,image_num_filename);
    [row, col, channel, number] = size(images);
    ln_exposure_time = log(e);
    
    choosed_point = zeros(point_num,2);
    [choosed_point] =  get_image_point (choosing_point_image,point_num);
    
    disp('Choose points on photo.');
    simages = zeros(point_num, 1, channel, number);
    for i = 1:number
        for j = 1:point_num
            simages(j,1,:,i) = images(choosed_point(j,1),choosed_point(j,2),:,i);
        end
    end
    
    disp('Alignment.');
    for i=1:number-1
        shift_param = zeros(1,2);
        init_param = zeros(1,2);
        shift_param = alignment(g_images(:,:,i),g_images(:,:,i+1),shift_bits,init_param);
        disp(i+1);
        disp(shift_param);
        images(:,:,:,i+1) = im_shift_vec3(images(:,:,:,i+1),shift_param);
        g_images(:,:,i+1) = im_shift(g_images(:,:,i+1),shift_param);
    end
    
    disp('Response curve by gsolve.');
    g = zeros(256, 3);
    lnE = zeros(point_num, 3);
    w = weightingFunction();
    w = w/max(w);

    for channel = 1:3
	pixel_data = reshape(simages(:,:,channel,:), point_num, number);
    [g(:,channel), lnE(:,channel)] = gsolve(pixel_data, ln_exposure_time, lambda, w);
    end
    
    disp('constructing HDR radiance map.');
    radiance_map = construct_radiance_map(images, g, ln_exposure_time, w);
    hdrwrite(radiance_map ,'radiance_map.hdr');
end