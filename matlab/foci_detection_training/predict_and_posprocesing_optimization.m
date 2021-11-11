clc;clear all;close all force;
addpath('../utils')

tmp_folder = '.';

load([tmp_folder '/final_net.mat'],'dlnet')
load([tmp_folder '/names.mat'],'files_test','files_valid','files_train')

data_path='../../../tmp/cell_lines';


data_chanels = {'a','b'};
matReaderData = @(x) matReader(x,'data',data_chanels,'norm_perc');
mask_chanels = {'a','b','ab'};
matReaderMask = @(x) matReader(x,'mask',mask_chanels,'norm_no');

in_layers = length(data_chanels);
out_layers = length(mask_chanels);


patchSize = [96 96 48];

tmp_folder2 = '../../../tmp_res_foci';

tmp_folder_valid = [tmp_folder2 '_test'];

files_valid_result = {};
for file_num = 1:length(files_valid)

    disp(['evaluation valid  '  num2str(file_num)  '/' num2str(length(files_valid))])

    file  = files_valid{file_num};
    data = matReaderData([file num2str(0)]);

    mask_predicted = predict_by_parts_foci_new(data,out_layers,dlnet,patchSize);

    results_name = replace( norm_path(file), norm_path(data_path), norm_path(tmp_folder_valid));

    results_path = fileparts(results_name);

    results_name = [results_path '/result.mat'];

    mkdir(results_path)

    save(results_name,'mask_predicted')

    files_valid_result = [files_valid_result,results_name];
end

tmp_folder_test = [tmp_folder2 '_test'];

files_test_result = {};
for file_num = 1:length(files_test)

    disp(['evaluation test  '  num2str(file_num)  '/' num2str(length(files_test))])

    file  = files_test{file_num};
    data = matReaderData([file num2str(0)]);

    mask_predicted = predict_by_parts_foci_new(data,out_layers,dlnet,patchSize);

    results_name = replace( norm_path(file), norm_path(data_path), norm_path(tmp_folder_test));

    results_path = fileparts(results_name);

    results_name = [results_path '/result.mat'];

    mkdir(results_path)

    save(results_name,'mask_predicted')

    files_test_result = [files_test_result,results_name];

end




T = optimizableVariable('T',[0.6,8.5]);
h = optimizableVariable('h',[0.1,9.9]);
d = optimizableVariable('d',[2,25]);

vars = [T,h,d];

for evaluate_index = 2:out_layers

    fun = @(x) -evaluate_detection_all(files_valid,files_valid_result,evaluate_index,matReaderMask,x.T,x.h,x.d);

    opt_results = bayesopt(fun,vars,'NumSeedPoints',5,'MaxObjectiveEvaluations',25,'UseParallel',false);


    x = opt_results.XAtMinObjective;


    [test_dice,results_points,gt_points] = evaluate_detection_all(files_test,files_test_result,evaluate_index,matReaderMask,x.T,x.h,x.d);


%     save([tmp_folder '/resutls_' mask_chanels{evaluate_index} '.mat'],'opt_results','test_dice','results_points','gt_points')

%     aa = 1;
%     save([tmp_folder '/test_dice_' mask_chanels{evaluate_index} '_' num2str(test_dice)  '.mat'],'aa')

    save([tmp_folder '/optimal_postprocessing_parameters_' mask_chanels{evaluate_index} '.mat'],'opt_results')
    disp(test_dice)
end