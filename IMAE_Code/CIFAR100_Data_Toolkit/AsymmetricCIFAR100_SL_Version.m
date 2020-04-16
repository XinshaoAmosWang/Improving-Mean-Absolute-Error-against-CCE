
% set noise rate (noise probabilities) and fixing the random seed.
rand_seed = 123; 
rng(rand_seed);
noise_prob = 0.2; 
save_path = 'SLAsymmetricTrainImageDataCell0.2.mat';

%%
load('train.mat');
shuffle_index = randperm(size(data,1));
data(shuffle_index, :) = data;
fine_labels(shuffle_index, :) = fine_labels;
coarse_labels(shuffle_index, :) = coarse_labels;
filenames(shuffle_index,:) = filenames;
%
labels = fine_labels; %0-99
class_num = 100;
width = 32;
height = 32;
depth = 3;


%folder = 'CIFARImages';
ImageDataCell = cell(size(data,1), 1);
for index = 1 : size(data, 1)
    img_data = data(index, :);
    img_data = reshape(img_data, width,height,depth);
    img_data = permute(img_data,[2,1,3]);% switch width and height to be regularised images
    %imshow(img_data);
    ImageDataCell{index} = img_data;
end

class_ids = zeros(size(data,1), 1);
superclass_num = size(unique(coarse_labels));
for sc = 1 : superclass_num
    super_label = sc-1;
    positions = coarse_labels==super_label;
    labels_subset = unique(fine_labels(positions));
    %Processing for  each superclass
    labelsToFlip = labels_subset( randperm(length(labels_subset), 2) );
    assert(labelsToFlip(1) ~= labelsToFlip(2));
    %
    positions_ofClassA =  fine_labels==labelsToFlip(1);
    positions_ofClassB =  fine_labels==labelsToFlip(2);
    %Fliping noise_prob of A to B, anb noise_prob of B to A
    %Simply first noise_prob since data is already shuffled
    sizeA = sum(positions_ofClassA);
    sizeB = sum(positions_ofClassB);
    labelsA = double(labelsToFlip(1)) * ones(sizeA, 1);
    labelsB = double(labelsToFlip(2)) * ones(sizeB, 1);
    labelsA(1:floor(noise_prob*sizeA) ) = labelsToFlip(2); % A->B
    labelsB(1:floor(noise_prob*sizeB) ) = labelsToFlip(1); % B->A 

    labels(positions_ofClassA ) = labelsA;
    labels(positions_ofClassB ) = labelsB;
end

actual_noise_rate = 1 - sum(labels ==  fine_labels)/size(data,1)
class_ids = labels;

save(save_path,...
        'ImageDataCell', ...
        'class_ids', ...
        '-v7.3');
    