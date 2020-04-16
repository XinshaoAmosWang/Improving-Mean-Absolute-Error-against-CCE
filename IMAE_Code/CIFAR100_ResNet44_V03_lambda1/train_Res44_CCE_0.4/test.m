%%
%pause(1*60*60*4.0)

addpath('../pre_pro_process');
addpath('../pre_pro_process/utils');
load( 'TestImageDataCell.mat' );

addpath ../../CaffeMex_UnifiedWeight_V01/matlab/
mainDir = '../';
modelDir = 'deploy_prototxts';









param.gpu_id = 2;
param.test_batch_size = 64;
param.test_net_file = fullfile(mainDir, modelDir, 'test_Res44.prototxt');
param.save_start = 1000;
param.save_interval = 1000;
param.train_maxiter = 30000;








param.save_model_file = 'checkpoints';
param.save_model_name = 'checkpoints_iter';
split_index = 1;
param.use_gpu = 1;
gpuDevice(param.gpu_id + 1);

param.crop_padding = 4;
param.force_square_size = 32+4;
param.cropped_size = 32;

train_x_axis=[];
train_y_axis=[];

for iter = param.save_start : param.save_interval : param.train_maxiter
        cur_path = pwd;
        caffe.reset_all;
        caffe.set_mode_gpu();
        caffe.set_device(param.gpu_id);
        caffe.init_log(fullfile(cur_path, 'log'));

        model_path = strcat(param.save_model_file, num2str(split_index),...
                                                '/', param.save_model_name, '_', num2str(iter), '.caffemodel');
        net = caffe.get_net(param.test_net_file, model_path, 'test');

        record_file = 'accuracy.txt';
        test_ori_center_crop_v01;

        train_x_axis = [train_x_axis, iter];
        train_y_axis = [train_y_axis, accuracy];
end

mat_path = strcat('accuracy', num2str(iter), '.mat');
save(mat_path, 'train_x_axis', 'train_y_axis');

plot(train_x_axis, train_y_axis);
saveas(gcf,'accuracy_curve.png')

exit;
