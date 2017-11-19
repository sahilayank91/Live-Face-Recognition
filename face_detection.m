clc
clear
close all

faceDetector = vision.CascadeObjectDetector;
shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[0 255 255]);

Im = imread('people2.jpg');
I = imresize(Im,[400,NaN]);
imshow(I);shg;

bbox = step(faceDetector, I);
% Draw boxes around detected faces and display results
I_faces = step(shapeInserter, I, int32(bbox));
imshow(I_faces), title('Detected faces');

for i = 1:size(bbox,1)
    J = imcrop(I,bbox(i,:));
    %figure(3),subplot(2,2,i);imshow(J);
    imshow(J);
end