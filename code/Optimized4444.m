%% ========================================================================
% SCRIPT نهایی با بهبودهای جزئی بر اساس تحلیل پیشرفته
% تغییرات:
% 1. افزایش نرخ Dropout برای مقابله بهتر با Overfitting.
% 2. افزایش ValidationPatience برای بهبود فرآیند Early Stopping.
% ========================================================================
clc; clear; close all;
%% --- بخش ۱: تنظیمات اصلی و پارامترها ---
cfg = struct();
% --- مسیرها ---
cfg.base_data_path = 'D:\bci-dataset\EEG_ConvertedData'; % !!! مسیر را به دیتاست خود تغییر دهید !!!
cfg.results_dir = 'Results_Final_Optimized4444';
% --- پارامترهای آزمایش ---
cfg.subjects = 1:25; 
cfg.sessions = 1:3;
cfg.tasks_info = {'reaching', 'Reach'; 'multigrasp', 'Grasp'; 'Twist', 'Twist'};
cfg.k_fold = 10; 
cfg.cv_repetitions = 1; 
cfg.max_epochs = 100; 
cfg.minibatch_size = 64;
% --- پارامترهای پیش‌پردازش ---
cfg.sel_channels = [11,12,13,15,16,17,18,20,21,22,23,39,40,41,43,44,45,47,48,49];
cfg.fs_new = 250;
cfg.bandpass_freq = [8 30];
cfg.fbands = [8 13; 13 30];
% --- پارامترهای آموزش ---
cfg.learning_rate = 0.0005;
% --- آماده‌سازی پوشه نتایج ---
if ~exist(cfg.results_dir, 'dir'), mkdir(cfg.results_dir); end
%% --- بخش ۲: حلقه اصلی پردازش (بدون تغییر) ---
num_subjects = length(cfg.subjects);
num_sessions = length(cfg.sessions);
num_tasks = size(cfg.tasks_info, 1);
results = NaN(num_subjects, num_tasks, num_sessions, 2); 
all_Y_test_per_task = cell(1, num_tasks);
all_Y_pred_per_task = cell(1, num_tasks);
for i = 1:num_subjects
    sub_id = cfg.subjects(i);
    fprintf('<<<<<<<<<< Starting Subject %d >>>>>>>>>>\n', sub_id);
    for j = 1:num_sessions
        sess_id = cfg.sessions(j);
        for k = 1:num_tasks
            task_name_file = cfg.tasks_info{k, 1};
            file_name = sprintf('EEG_session%d_sub%d_%s_MI.mat', sess_id, sub_id, task_name_file);
            file_path = fullfile(cfg.base_data_path, file_name);
            if exist(file_path, 'file')
                fprintf('--- Running: Subject %d, Session %d, Task: %s ---\n', sub_id, sess_id, task_name_file);
                try
                    [mean_acc, std_acc, Y_test_all, Y_pred_all] = process_eeg_task(file_path, task_name_file, cfg);
                    results(i, k, j, 1) = mean_acc * 100; results(i, k, j, 2) = std_acc * 100;
                    fprintf('Final Result: Accuracy = %.2f%% (±%.2f)\n', results(i, k, j, 1), results(i, k, j, 2));
                    all_Y_test_per_task{k} = [all_Y_test_per_task{k}; Y_test_all];
                    all_Y_pred_per_task{k} = [all_Y_pred_per_task{k}; Y_pred_all];
                catch ME
                    fprintf('!!! Error processing file %s: %s\n', file_name, ME.message);
                    fprintf('!!! Error on line %d of file %s\n', ME.stack(1).line, ME.stack(1).name);
                    rethrow(ME);
                end
            else, fprintf('!!! File not found: %s\n', file_name); end
        end
    end
    fprintf('>>>>> Results saved up to subject %d. <<<<<\n', sub_id);
    save(fullfile(cfg.results_dir, 'intermediate_results.mat'), 'results', 'all_Y_test_per_task', 'all_Y_pred_per_task');
    fprintf('<<<<<<<<<< Finished Subject %d >>>>>>>>>>\n\n', sub_id);
end
%% --- بخش ۳: گزارش‌دهی و رسم نمودار (بدون تغییر) ---
display_and_save_results(results, cfg);
plot_and_save_charts(all_Y_test_per_task, all_Y_pred_per_task, results, cfg);
%% --- بخش ۴: تعریف توابع اصلی (Local Functions) ---
function [mean_acc, std_acc, Y_test_all, Y_pred_all] = process_eeg_task(file_path, task_name, cfg)
    [X, Y, C, num_classes, Nbands, trial_groups] = load_and_preprocess_data(file_path, task_name, cfg);
    [PLV_4d, POW_4d] = extract_features_graphcnn(X, cfg.fs_new, cfg.fbands);
    acc_all_reps = zeros(cfg.cv_repetitions, 1);
    Y_pred_all = categorical([]); Y_test_all = categorical([]);
    for rep = 1:cfg.cv_repetitions
        fprintf('--- Repetition %d/%d ---\n', rep, cfg.cv_repetitions);
        cv = cvpartition(trial_groups, 'KFold', cfg.k_fold);
        acc_per_fold = zeros(cfg.k_fold, 1);
        for fold = 1:cfg.k_fold
            trainIdx_logical = training(cv, fold); testIdx_logical = test(cv, fold);
            cv_train = cvpartition(sum(trainIdx_logical), 'HoldOut', 0.1);
            train_indices_numeric = find(trainIdx_logical);
            mainTrainIdx_numeric = train_indices_numeric(training(cv_train));
            validationIdx_numeric = train_indices_numeric(test(cv_train));
            dsPLV_val = arrayDatastore(PLV_4d(:,:,:,validationIdx_numeric), 'IterationDimension', 4);
            dsPOW_val = arrayDatastore(POW_4d(:,:,:,validationIdx_numeric), 'IterationDimension', 4);
            dsY_val = arrayDatastore(Y(validationIdx_numeric));
            dsValidation = combine(dsPLV_val, dsPOW_val, dsY_val);
            options = trainingOptions('adam', 'MaxEpochs', cfg.max_epochs, 'MiniBatchSize', cfg.minibatch_size, 'Shuffle', 'every-epoch', 'InitialLearnRate', cfg.learning_rate, 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 15, 'Verbose', false, 'Plots', 'none', 'ValidationData', dsValidation, 'ValidationFrequency', 10, 'ValidationPatience', 15); % *** بهبود: افزایش Patience
            lgraph = create_cnn_with_cbam_attention(C, Nbands, num_classes);
            dsPLV_train = arrayDatastore(PLV_4d(:,:,:,mainTrainIdx_numeric), 'IterationDimension', 4);
            dsPOW_train = arrayDatastore(POW_4d(:,:,:,mainTrainIdx_numeric), 'IterationDimension', 4);
            dsY_train = arrayDatastore(Y(mainTrainIdx_numeric));
            dsTrain = combine(dsPLV_train, dsPOW_train, dsY_train);
            net = trainNetwork(dsTrain, lgraph, options);
            dsPLV_test = arrayDatastore(PLV_4d(:,:,:,testIdx_logical), 'IterationDimension', 4);
            dsPOW_test = arrayDatastore(POW_4d(:,:,:,testIdx_logical), 'IterationDimension', 4);
            dsTest = combine(dsPLV_test, dsPOW_test);
            Y_pred = classify(net, dsTest);
            acc_per_fold(fold) = mean(Y_pred == Y(testIdx_logical));
            Y_pred_all = [Y_pred_all; Y_pred]; 
            Y_test_all = [Y_test_all; Y(testIdx_logical)];
        end
        acc_all_reps(rep) = mean(acc_per_fold);
    end
    mean_acc = mean(acc_all_reps); 
    std_acc = std(acc_all_reps);
end
%% --- بخش ۵: تعریف معماری شبکه با توجه CBAM ---
function lgraph = create_cnn_with_cbam_attention(num_nodes, num_bands, num_classes)
    lgraph = layerGraph();
    
    inputPLV = imageInputLayer([num_nodes num_nodes num_bands], 'Name', 'inputPLV', 'Normalization', 'zscore');
    inputPOW = imageInputLayer([num_nodes num_bands 1], 'Name', 'inputPOW', 'Normalization', 'zscore');
    lgraph = addLayers(lgraph, inputPLV);
    lgraph = addLayers(lgraph, inputPOW);
    
    % --- PLV Branch ---
    num_channels_plv = 32;
    plv_conv_branch = [
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_plv1')
        batchNormalizationLayer('Name', 'bn_plv1')
        reluLayer('Name', 'relu_plv1')
        averagePooling2dLayer(2, 'Stride', 2, 'Name', 'pool_plv1')
        convolution2dLayer(3, num_channels_plv, 'Padding', 'same', 'Name', 'conv_plv2')
        batchNormalizationLayer('Name', 'bn_plv2')
        reluLayer('Name', 'relu_plv2')
    ];
    lgraph = addLayers(lgraph, plv_conv_branch);
    lgraph = connectLayers(lgraph, 'inputPLV', 'conv_plv1');
    [lgraph, plv_attention_out] = add_cbam_block(lgraph, 'relu_plv2', num_channels_plv, 'plv');
    plv_out_branch = [
        globalAveragePooling2dLayer('Name', 'gap_plv')
        flattenLayer('Name', 'flatten_plv')
        fullyConnectedLayer(64, 'Name', 'fc_plv')
    ];
    lgraph = addLayers(lgraph, plv_out_branch);
    lgraph = connectLayers(lgraph, plv_attention_out, 'gap_plv');
    
    % --- POW Branch ---
    num_channels_pow = 32;
    pow_conv_branch = [
        convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv_pow1')
        batchNormalizationLayer('Name', 'bn_pow1')
        reluLayer('Name', 'relu_pow1')
        averagePooling2dLayer(2, 'Stride', 2, 'Name', 'pool_pow1')
        convolution2dLayer([3 3], num_channels_pow, 'Padding', 'same', 'Name', 'conv_pow2')
        batchNormalizationLayer('Name', 'bn_pow2')
        reluLayer('Name', 'relu_pow2')
    ];
    lgraph = addLayers(lgraph, pow_conv_branch);
    lgraph = connectLayers(lgraph, 'inputPOW', 'conv_pow1');
    [lgraph, pow_attention_out] = add_cbam_block(lgraph, 'relu_pow2', num_channels_pow, 'pow');
    pow_out_branch = [
        globalAveragePooling2dLayer('Name', 'gap_pow')
        flattenLayer('Name', 'flatten_pow')
        fullyConnectedLayer(64, 'Name', 'fc_pow')
    ];
    lgraph = addLayers(lgraph, pow_out_branch);
    lgraph = connectLayers(lgraph, pow_attention_out, 'gap_pow');
    
    % --- Fusion and Classification Branch ---
    fusion_layers = [
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(128, 'Name', 'fc_fusion1')
        reluLayer('Name', 'relu_fusion')
        dropoutLayer(0.6, 'Name', 'dropout') % *** بهبود: افزایش Dropout
        fullyConnectedLayer(num_classes, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classification')
    ];
    lgraph = addLayers(lgraph, fusion_layers);
    lgraph = connectLayers(lgraph, 'fc_plv', 'concat/in1');
    lgraph = connectLayers(lgraph, 'fc_pow', 'concat/in2');
end
function [lgraph, final_out_name] = add_cbam_block(lgraph, input_name, num_channels, prefix)
    reduction_ratio = 4;
    ca_avg_pool = globalAveragePooling2dLayer('Name', [prefix, '_ca_avg_pool']);
    ca_max_pool = globalMaxPooling2dLayer('Name', [prefix, '_ca_max_pool']);
    lgraph = addLayers(lgraph, ca_avg_pool);
    lgraph = addLayers(lgraph, ca_max_pool);
    lgraph = connectLayers(lgraph, input_name, [prefix, '_ca_avg_pool']);
    lgraph = connectLayers(lgraph, input_name, [prefix, '_ca_max_pool']);
    shared_mlp = [fullyConnectedLayer(round(num_channels / reduction_ratio), 'Name', [prefix, '_ca_mlp_fc1']), reluLayer('Name', [prefix, '_ca_mlp_relu']), fullyConnectedLayer(num_channels, 'Name', [prefix, '_ca_mlp_fc2'])];
    lgraph = addLayers(lgraph, shared_mlp);
    lgraph = connectLayers(lgraph, [prefix, '_ca_avg_pool'], [prefix, '_ca_mlp_fc1']);
    shared_mlp_max = [fullyConnectedLayer(round(num_channels / reduction_ratio), 'Name', [prefix, '_ca_mlp_fc1_max']), reluLayer('Name', [prefix, '_ca_mlp_relu_max']), fullyConnectedLayer(num_channels, 'Name', [prefix, '_ca_mlp_fc2_max'])];
    lgraph = addLayers(lgraph, shared_mlp_max);
    lgraph = connectLayers(lgraph, [prefix, '_ca_max_pool'], [prefix, '_ca_mlp_fc1_max']);
    ca_add = additionLayer(2, 'Name', [prefix, '_ca_add']);
    lgraph = addLayers(lgraph, ca_add);
    lgraph = connectLayers(lgraph, [prefix, '_ca_mlp_fc2'], [prefix, '_ca_add/in1']);
    lgraph = connectLayers(lgraph, [prefix, '_ca_mlp_fc2_max'], [prefix, '_ca_add/in2']);
    ca_sigmoid = sigmoidLayer('Name', [prefix, '_ca_sigmoid']);
    lgraph = addLayers(lgraph, ca_sigmoid);
    lgraph = connectLayers(lgraph, [prefix, '_ca_add'], [prefix, '_ca_sigmoid']);
    ca_scale = multiplicationLayer(2, 'Name', [prefix, '_ca_scale']);
    lgraph = addLayers(lgraph, ca_scale);
    lgraph = connectLayers(lgraph, input_name, [prefix, '_ca_scale/in1']);
    lgraph = connectLayers(lgraph, [prefix, '_ca_sigmoid'], [prefix, '_ca_scale/in2']);
    channel_attention_out = [prefix, '_ca_scale'];
    sa_conv = convolution2dLayer(7, 1, 'Padding', 'same', 'Name', [prefix, '_sa_conv']);
    sa_sigmoid = sigmoidLayer('Name', [prefix, '_sa_sigmoid']);
    lgraph = addLayers(lgraph, sa_conv);
    lgraph = addLayers(lgraph, sa_sigmoid);
    lgraph = connectLayers(lgraph, channel_attention_out, [prefix, '_sa_conv']);
    lgraph = connectLayers(lgraph, [prefix, '_sa_conv'], [prefix, '_sa_sigmoid']);
    sa_scale = multiplicationLayer(2, 'Name', [prefix, '_sa_scale']);
    lgraph = addLayers(lgraph, sa_scale);
    lgraph = connectLayers(lgraph, channel_attention_out, [prefix, '_sa_scale/in1']);
    lgraph = connectLayers(lgraph, [prefix, '_sa_sigmoid'], [prefix, '_sa_scale/in2']);
    final_out_name = [prefix, '_sa_scale'];
end
%% --- بخش ۶: توابع کمکی (بدون تغییر) ---
function [X, Y, C, num_classes, Nbands, trial_groups] = load_and_preprocess_data(file_path, task_name, cfg)
    S=load(file_path); mrk=S.mrk; C=length(cfg.sel_channels);
    cnt_raw=zeros(length(S.ch1),60);
    for ii=1:60, chname=['ch',num2str(ii)]; if isfield(S,chname), cnt_raw(:,ii)=double(S.(chname)); end, end
    cnt_selected = cnt_raw(:, cfg.sel_channels);
    mean_signal = mean(cnt_selected, 2); cnt_car = cnt_selected - mean_signal;
    cnt_resampled = resample(cnt_car, cfg.fs_new, mrk.fs);
    mrk.pos = round(mrk.pos * (cfg.fs_new / mrk.fs));
    [bN,aN]=iirnotch(60/(cfg.fs_new/2),(60/(cfg.fs_new/2))/35); cnt_notch=filtfilt(bN,aN,cnt_resampled);
    [bB,aB]=butter(4,cfg.bandpass_freq/(cfg.fs_new/2),'bandpass'); cnt_bp=filtfilt(bB,aB,cnt_notch);
    [~,labels]=max(mrk.y,[],1);
    trial_len=4*cfg.fs_new; win_len=2*cfg.fs_new; stride=0.5*cfg.fs_new;
    X_all=[]; Y_all=[]; trial_groups = [];
    for i=1:numel(mrk.pos)
        start=mrk.pos(i)+round(0.5*cfg.fs_new); stop=start+trial_len-1;
        if stop>size(cnt_bp,1), continue; end
        trial_data=cnt_bp(start:stop,:);
        for s=1:stride:(size(trial_data,1)-win_len+1)
            X_all=cat(3,X_all,trial_data(s:s+win_len-1,:)); 
            Y_all=[Y_all; labels(i)];
            trial_groups = [trial_groups; i];
        end
    end
    switch task_name, case 'multigrasp',rc=4; case 'Twist',rc=3; case 'reaching',rc=7; end
    keep=Y_all~=rc; 
    X=X_all(:,:,keep); 
    Y=categorical(Y_all(keep));
    trial_groups = trial_groups(keep);
    num_classes = numel(categories(Y));
    Nbands = size(cfg.fbands, 1);
end
function [PLV_matrices, POW_matrices] = extract_features_graphcnn(X_trials, fs, fbands)
    [~,C,N]=size(X_trials); num_bands=size(fbands,1);
    PLV_matrices=zeros(C,C,num_bands,N,'like',X_trials); 
    POW_matrices=zeros(C,num_bands,1,N,'like',X_trials);
    [B,A]=design_subband_filters(fs,fbands);
    if isempty(gcp('nocreate')), parpool; end
    parfor n=1:N
        trial_n=X_trials(:,:,n);
        for b=1:num_bands
            trial_band=filtfilt(B{b},A{b},trial_n);
            POW_matrices(:,b,1,n)=bandpower(trial_band);
            phase=angle(hilbert(trial_band));
            plv_b=zeros(C,C,'like',X_trials);
            for i=1:C, for j=i+1:C
                d_phase=phase(:,i)-phase(:,j); 
                plv_b(i,j)=abs(mean(exp(1i*d_phase))); 
                plv_b(j,i)=plv_b(i,j);
            end, end
            PLV_matrices(:,:,b,n)=plv_b;
        end
    end
end
function [B, A] = design_subband_filters(fs,fbands)
    nB=size(fbands,1); B=cell(nB,1); A=cell(nB,1);
    for b=1:nB, [B{b},A{b}]=butter(6,fbands(b,:)/(fs/2),'bandpass'); end
end
%% --- بخش ۷: توابع گزارش‌دهی (بدون تغییر) ---
function display_and_save_results(results, cfg)
    num_sessions = length(cfg.sessions);
    num_tasks = size(cfg.tasks_info, 1);
    num_subjects = length(cfg.subjects);
    fprintf('\n\n=================================== Final Results ===================================\n');
    header_line1 = 'Participant |';
    header_line2 = '            |';
    for j = 1:num_sessions
        header_line1 = [header_line1, sprintf(' %-28s|', ['Session ' num2str(j) ' (mean (±SD))'])];
        task_headers = '';
        for k = 1:num_tasks
            task_headers = [task_headers, sprintf('%-12s ', cfg.tasks_info{k, 2})];
        end
        header_line2 = [header_line2, ' ', task_headers, '|'];
    end
    results_table_path = fullfile(cfg.results_dir, 'results_table.txt');
    fileID = fopen(results_table_path, 'w', 'n', 'UTF-8');
    fprintf('%s\n', header_line1);
    fprintf('%s\n', header_line2);
    fprintf('%s\n', repmat('-', 1, strlength(header_line1)));
    fprintf(fileID, '%s\n%s\n%s\n', header_line1, header_line2, repmat('-', 1, strlength(header_line1)));
    for i = 1:num_subjects
        sub_id = cfg.subjects(i);
        line_str = sprintf('%-11d |', sub_id);
        for j = 1:num_sessions
            line_str = [line_str, ' '];
            for k = 1:num_tasks
                mean_val = results(i, k, j, 1);
                std_val = results(i, k, j, 2);
                if isnan(mean_val)
                    line_str = [line_str, sprintf('%-12s ', 'N/A')];
                else
                    line_str = [line_str, sprintf('%.2f(±%.2f) ', mean_val, std_val)];
                end
            end
            line_str = [line_str, '|'];
        end
        fprintf('%s\n', line_str);
        fprintf(fileID, '%s\n', line_str);
    end
    fprintf('%s\n', repmat('-', 1, strlength(header_line1)));
    fprintf(fileID, '%s\n', repmat('-', 1, strlength(header_line1)));
    mean_line = sprintf('%-11s |', 'Mean');
    for j = 1:num_sessions
        mean_line = [mean_line, ' '];
        for k = 1:num_tasks
            mean_of_means = nanmean(results(:, k, j, 1));
            std_of_means = nanstd(results(:, k, j, 1));
            mean_line = [mean_line, sprintf('%.2f(±%.2f) ', mean_of_means, std_of_means)];
        end
        mean_line = [mean_line, '|'];
    end
    fprintf('%s\n', mean_line);
    fprintf(fileID, '%s\n', mean_line);
    fclose(fileID);
    fprintf('\n\nResults table successfully saved to "%s"\n', results_table_path);
end
function plot_and_save_charts(all_Y_test_per_task, all_Y_pred_per_task, results, cfg)
    num_tasks = size(cfg.tasks_info, 1);
    num_subjects = length(cfg.subjects);
    num_sessions = length(cfg.sessions);
    fprintf('\n\n========================= Generating Overall Confusion Matrices =========================\n');
    for k = 1:num_tasks
        task_name_disp = cfg.tasks_info{k, 2};
        if ~isempty(all_Y_test_per_task{k})
            fig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
            cm = confusionchart(all_Y_test_per_task{k}, all_Y_pred_per_task{k}, 'Title', sprintf('Overall CM for Task: %s', task_name_disp), 'FontSize', 12);
            cm_filename = fullfile(cfg.results_dir, sprintf('Overall_CM_Task_%s.png', task_name_disp));
            saveas(fig, cm_filename); close(fig);
            fprintf('Overall confusion matrix for task %s saved successfully.\n', task_name_disp);
        end
    end
    fprintf('\n\n========================== Generating Performance Analysis Charts ==========================\n');
    fig = figure('Visible', 'off', 'Position', [100, 100, 900, 600]);
    if num_sessions > 1
        mean_results_per_session_task = squeeze(nanmean(results(:,:,:,1), 1));
        bar(mean_results_per_session_task);
        legend(arrayfun(@(x) ['Session ' num2str(x)], 1:num_sessions, 'UniformOutput', false), 'Location', 'northwest');
    else
        mean_results_per_session_task = squeeze(nanmean(results(:,:,:,1), 1))';
        bar(mean_results_per_session_task);
    end
    xticklabels(cfg.tasks_info(:, 2)); xlabel('Task'); ylabel('Mean Accuracy (%)');
    title('Comparison of Mean Accuracy by Task and Session');
    grid on; set(gca, 'FontSize', 12); saveas(fig, fullfile(cfg.results_dir, 'Task_Accuracy_BarPlot.png')); close(fig);
    fprintf('Task accuracy bar plot saved successfully.\n');
    fig = figure('Visible', 'off', 'Position', [100, 100, 900, 600]);
    data_for_boxplot = reshape(results(:, :, :, 1), num_subjects * num_sessions, num_tasks);
    boxplot(data_for_boxplot, 'Labels', cfg.tasks_info(:, 2)); xlabel('Task'); ylabel('Accuracy (%)');
    title('Accuracy Distribution Across Subjects for Each Task'); grid on; set(gca, 'FontSize', 12);
    saveas(fig, fullfile(cfg.results_dir, 'Task_Accuracy_BoxPlot.png')); close(fig);
    fprintf('Task accuracy box plot saved successfully.\n');
    if num_sessions > 1
        fig = figure('Visible', 'off', 'Position', [100, 100, 900, 600]);
        line_data = squeeze(nanmean(results(:,:,:,1), 1))';
        plot(1:num_sessions, line_data, '-o', 'LineWidth', 2, 'MarkerSize', 8); xticks(1:num_sessions); xlabel('Session');
        ylabel('Mean Accuracy (%)'); title('Mean Accuracy Trend Across Sessions'); legend(cfg.tasks_info(:, 2), 'Location', 'best');
        grid on; set(gca, 'FontSize', 12); saveas(fig, fullfile(cfg.results_dir, 'Session_Accuracy_LinePlot.png')); close(fig);
        fprintf('Session accuracy line plot saved successfully.\n');
    end
end
