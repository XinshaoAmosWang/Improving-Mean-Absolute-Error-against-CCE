

load('train.mat');
labels = fine_labels;
class_num = 100;
width = 32;
height = 32;
depth = 3;

% set noise rate (noise probabilities) and fixing the random seed.
rand_seed = 123; rng(rand_seed);
noise_prob = 0.2; 
save_path = 'TrainImageDataCell0.2.mat';

%folder = 'CIFARImages';
ImageDataCell = cell(size(data,1), 1);
class_ids = zeros(size(data,1), 1);
for index = 1 : size(data, 1)
    img_data = data(index, :);
    img_data = reshape(img_data, width,height,depth);
    img_data = permute(img_data,[2,1,3]);% switch width and height to be regularised images
    %imshow(img_data);
    
    if rand() < noise_prob %label is corrupted
        true_label = labels(index);
        labels_subset = (1:class_num) - 1;
        labels_subset(true_label+1)=[];  % remove the true label in the labels_subset
        img_label = labels_subset( randperm(class_num-1, 1) );
    else % original label
        img_label = labels(index);
    end
    
    ImageDataCell{index} = img_data;
    class_ids(index) = img_label;
end


save(save_path,...
        'ImageDataCell', ...
        'class_ids', ...
        '-v7.3');
    