function [image_probs] = forward_ori_center_crop_v01(net, ImageDataCell, batch_size, ...
                                            crop_padding, force_square_size, cropped_size)
%forward_images_data 
%Summary of this function goes here
%   1. Input
%       net: the trained model
%       ImageDataCell: image data cell
%       batch_size: the maximum input for each forward
%   2. Process
%       a. form batch
%       b. forward
%       c. extract output probs
%   3. Output: (n, dim)
image_probs = [];

num_img = size(ImageDataCell, 1);
num_batch = floor(num_img/batch_size);

net.blobs('data').reshape([cropped_size cropped_size 3 batch_size]);

for bt = 1 : num_batch
    % form batch
    batch_data = zeros(cropped_size, cropped_size, 3, batch_size, 'single');
    for ind = 1 : batch_size
        ind_img = (bt-1)*batch_size + ind;
        batch_data(:,:,:, ind) = process_imgdataint8(ImageDataCell{ind_img, 1}, ...
                                                           crop_padding, force_square_size, ...
                                                           cropped_size, false);
    end
    % set_data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract probs
    image_probs = [image_probs; (squeeze(net.blobs('probs').get_data))'];
end

% remained images
if  mod(num_img, batch_size) ~= 0
    remained_images = ImageDataCell(num_batch*batch_size + 1 : end, :);
    num_img_remained = size(remained_images, 1);

    batch_data = zeros(cropped_size, cropped_size, 3, num_img_remained, 'single');
    net.blobs('data').reshape([cropped_size cropped_size 3 num_img_remained]);

    for ind = 1 : num_img_remained
        batch_data(:,:,:, ind) = process_imgdataint8(remained_images{ind, 1}, ...
                                                           crop_padding, force_square_size, ...
                                                           cropped_size, false);
    end

    % set data, forward
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    % extract probs
    image_probs = [image_probs; (squeeze(net.blobs('probs').get_data))'];
end


end

