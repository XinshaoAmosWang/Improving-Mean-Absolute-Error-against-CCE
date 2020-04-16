% Workpace parameters:
% net:
% param.test_batch_size: 

% data_cell
% class_ids

test_probs = forward_ori_center_crop_v01(net, ImageDataCell, param.test_batch_size, ...
							 param.crop_padding, param.force_square_size, param.cropped_size );
test_num = size(test_probs, 1);

[~, preds] = max(test_probs');
preds=preds'-1;

accuracy = sum(class_ids==preds)/test_num;



fin = fopen(record_file, 'a');
fprintf(fin, 'iter: %d, accuracy: %.4f\n', ...
                                iter, accuracy);
fprintf('iter: %d, accuracy: %.4f\n', ...
                                iter, accuracy);
fclose(fin);


