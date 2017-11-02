

for i=1:10
    ss  = imread(strcat(num2str(i),'.jpeg'));
    imshow(ss);
    frame = rgb2gray(ss);
    imresize(frame,[400 600]);
    imwrite(imresize(frame, [112 92]), strcat(num2str(i),'.pgm'));
end

