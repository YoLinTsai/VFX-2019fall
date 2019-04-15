function [loc_list] = get_image_point (image,point_num)

figure('name','Click to set location');
imshow(image);
loc_list = zeros(point_num,2);
points = 0;
[row, col, channel] = size(image);

while points < point_num 
    [y, x] = ginput(1);
    check=1;
    for j=1:points
        if x == loc_list(j,1) && y == loc_list(j,2)
            disp('This point has been choosen!');
            check=0;
        end
        if x<1 || x>row || y<1 || y>col
            disp('This point is out of range!');
            check = 0;
        end
    end
    if check == 1
         points=points+1;
        loc_list(points,1)=int32(x);
        loc_list(points,2)=int32(y);
    end
end

close all;
end
