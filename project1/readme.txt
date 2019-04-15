將img_data.txt和img_num.txt放到和matlab library同一層資料夾

data的形式為

資料夾\名稱 曝光時間(s)
stair\0001.jpg 8
stair\0002.jpg 4
stair\0003.jpg 2
stair\0004.jpg 1
stair\0005.jpg 0.5
stair\0006.jpg 0.25
stair\0007.jpg 0.125
stair\0008.jpg 0.05

img_num.txt的格式為

9

function format

function radiance_map = myhdr(img_data_filename,image_num_filename,point_num,shift_bits)

function ldr = r_tonemapping(hdr)