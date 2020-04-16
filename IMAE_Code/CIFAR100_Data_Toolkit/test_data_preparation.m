

load('test.mat')
labels = fine_labels;
class_num = 100;
width = 32;
height = 32;
depth = 3;


%folder = 'CIFARImages';
ImageDataCell = cell(size(data,1), 1);
class_ids = zeros(size(data,1), 1);
for index = 1 : size(data, 1)
    img_data = data(index, :);
    img_data = reshape(img_data, width,height,depth);
    img_data = permute(img_data,[2,1,3]);% switch width and height to be regularised images
    %imshow(img_data);
    
    img_label = labels(index);
    
    ImageDataCell{index} = img_data;
    class_ids(index) = img_label;
end


save_path = 'TestImageDataCell.mat';
save(save_path,...
        'ImageDataCell', ...
        'class_ids', ...
        '-v7.3');
    