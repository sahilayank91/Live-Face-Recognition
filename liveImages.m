url = 'http://172.17.21.189:8080/shot.jpg';
ss  = imread(url);
fh = image(ss);
for i=1:10
    ss  = imread(url);
    I = rgb2gray(ss);
    imwrite(imresize(frame, [112 92]), strcat(num2str(i),'.pgm'));
    imshow(I);
    delay(5);
end