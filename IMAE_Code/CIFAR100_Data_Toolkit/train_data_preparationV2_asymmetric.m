

load('train.mat');
labels = fine_labels; %0-99
class_num = 100;
width = 32;
height = 32;
depth = 3;

% set noise rate (noise probabilities) and fixing the random seed.
rand_seed = 123; rng(rand_seed);
noise_prob = 0.2; 
save_path = 'AsymmetricTrainImageDataCell0.2.mat';

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
        super_label = coarse_labels(index);
        
        % To find labels pool for corruption
        labels_subset = unique(fine_labels(coarse_labels==super_label));
        true_label_index = find(labels_subset==true_label);
        labels_subset(true_label_index) = []; % remove the true label in the labels_subset
        
        % choose one randomly from the subset/pool
        img_label = labels_subset( randperm(length(labels_subset), 1) );
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
    