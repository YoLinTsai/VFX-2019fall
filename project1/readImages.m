function [choosing_point_image, g_images, images, exposure_time] = readImages(file_data_name,file_num_name)
    images = [];
    exposure_time = [];
    g_images = [];
    
    [image_num]=textread(file_num_name,'%d');
    [file_name,exposure_time]=textread(file_data_name,'%s %f');

    %disp(exposure_time);
    %disp(file_name(1));
    
    info = imfinfo(file_name{1});
    number = length(file_name);
    images = zeros(info.Height, info.Width, info.NumberOfSamples, image_num);
    g_images = zeros(info.Height, info.Width, image_num);
    
    for i = 1:image_num
        temp_img = imread(file_name{i});
        images(:,:,:,i) = temp_img;
        g_images(:,:,i) = rgb2gray(temp_img);
    end
    
    choosing_point_image = imread(file_name{1});
end