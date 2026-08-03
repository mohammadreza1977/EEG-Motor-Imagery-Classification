% --- MATLAB Code to Calculate Mean Accuracy for the SE-Block Model ---

% Step 1: Define the data matrix from the new results table.
% Each row is a participant (25 total).
% Columns 1-3: Session 1 (Reach, Grasp, Twist)
% Columns 4-6: Session 2 (Reach, Grasp, Twist)
% Columns 7-9: Session 3 (Reach, Grasp, Twist)

results_se_block = [
    28.13, 40.73, 64.20,  25.53, 43.20, 68.60,  29.33, 55.47, 68.60;
    29.27, 40.93, 58.00,  22.87, 41.47, 62.20,  25.07, 42.13, 52.00;
    23.13, 51.20, 63.60,  22.07, 44.27, 64.80,  24.47, 47.47, 55.20;
    28.33, 45.47, 63.00,  28.80, 57.60, 68.40,  35.27, 54.67, 64.20;
    27.53, 41.93, 62.00,  23.20, 47.07, 60.20,  24.13, 50.80, 66.40;
    30.53, 46.80, 65.60,  24.53, 45.47, 55.20,  27.13, 43.47, 61.60;
    26.67, 49.60, 60.60,  27.67, 46.93, 66.80,  26.80, 46.53, 55.00;
    30.80, 35.87, 61.80,  26.93, 48.40, 60.40,  26.67, 45.07, 63.20;
    24.73, 55.60, 61.20,  22.87, 42.27, 64.40,  25.00, 46.27, 62.00;
    30.47, 42.27, 65.40,  30.80, 49.07, 62.00,  29.33, 46.00, 66.60;
    29.53, 48.40, 60.80,  34.40, 50.53, 61.80,  25.87, 48.40, 66.60;
    29.13, 41.60, 60.80,  26.47, 46.00, 61.00,  22.27, 49.73, 65.40;
    24.67, 48.20, 63.20,  26.33, 43.07, 62.00,  21.87, 49.07, 63.00;
    27.93, 48.80, 66.00,  24.53, 48.27, 61.00,  28.13, 46.00, 58.40;
    24.27, 52.20, 60.20,  27.20, 39.87, 65.40,  24.20, 40.40, 63.40;
    29.00, 50.25, 65.00,  26.60, 44.13, 50.60,  28.93, 48.80, 65.60;
    24.73, 44.67, 63.20,  21.60, 43.20, 67.60,  22.60, 43.20, 63.80;
    23.73, 48.20, 64.80,  25.27, 47.07, 50.00,  23.73, 41.20, 63.20;
    26.40, 49.13, 52.40,  25.27, 44.27, 61.00,  24.33, 44.27, 59.60;
    22.33, 48.80, 58.40,  25.40, 47.60, 64.80,  24.07, 46.40, 62.00;
    24.27, 46.00, 63.40,  25.27, 43.73, 64.00,  24.47, 46.33, 61.20;
    26.03, 49.73, 55.40,  22.13, 45.20, 72.60,  24.33, 41.33, 60.60;
    31.33, 51.93, 62.60,  30.13, 40.67, 70.20,  23.93, 39.07, 59.20;
    25.07, 47.87, 64.40,  22.87, 54.13, 59.20,  21.40, 45.20, 60.40;
    22.93, 44.27, 62.40,  23.47, 53.20, 58.00,  28.67, 48.67, 64.80
];

% Step 2: Calculate the overall mean of all data points.
overall_mean_se = mean(results_se_block(:));

% Step 3: Display the final result.
fprintf('Overall Mean Accuracy for (CNN + SE-Block) is: %.2f%%\n', overall_mean_se);

% --- Additional Calculations (Optional) ---

% Calculate the average for each task across all sessions.
mean_reach_se = mean(mean(results_se_block(:, [1, 4, 7])));
mean_grasp_se = mean(mean(results_se_block(:, [2, 5, 8])));
mean_twist_se = mean(mean(results_se_block(:, [3, 6, 9])));

fprintf('\n--- Further Analysis ---\n');
fprintf('Average accuracy for the Reaching task: %.2f%%\n', mean_reach_se);
fprintf('Average accuracy for the Grasping task: %.2f%%\n', mean_grasp_se);
fprintf('Average accuracy for the Twisting task: %.2f%%\n', mean_twist_se);