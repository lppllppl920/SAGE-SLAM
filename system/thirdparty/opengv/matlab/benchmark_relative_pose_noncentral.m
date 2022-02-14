%% Reset everything

clear all;
clc;
close all;
addpath('helpers');

%% Configure the benchmark

% noncentral case
cam_number = 4;
% Getting 10 points, and testing all algorithms with the respective number of points
pt_number = 50;
% noise test, so no outliers
outlier_fraction = 0.0;
% repeat 5000 tests per noise level
iterations = 5000;

% The algorithms we want to test
algorithms = { 'seventeenpt'; 'rel_nonlin_noncentral'; 'rel_nonlin_noncentral' };
% This defines the number of points used for every algorithm
indices = { [1:1:17]; [1:1:17]; [1:1:50] };
% The name of the algorithms in the final plots
names = { '17pt'; 'nonlin. opt. (17pts)'; 'nonlin. opt. (50pts)' };

% The maximum noise to analyze
max_noise = 5.0;
% The step in between different noise levels
noise_step = 0.1;

%% Run the benchmark

%prepare the overall result arrays
number_noise_levels = max_noise / noise_step + 1;
num_algorithms = size(algorithms,1);
mean_rotation_errors = zeros(num_algorithms,number_noise_levels);
median_rotation_errors = zeros(num_algorithms,number_noise_levels);
mean_position_errors = zeros(num_algorithms,number_noise_levels);
median_position_errors = zeros(num_algorithms,number_noise_levels);
noise_levels = zeros(1,number_noise_levels);

%Run the experiment
for n=1:number_noise_levels

    noise = (n - 1) * noise_step;
    noise_levels(1,n) = noise;
    display(['Analyzing noise level: ' num2str(noise)])
    
    rotation_errors = zeros(num_algorithms,iterations);
    position_errors = zeros(num_algorithms,iterations);
    
    counter = 0;
    
    for i=1:iterations
        
        % generate experiment        
        [v1,v2,t,R] = create2D2DExperiment(pt_number,cam_number,noise,outlier_fraction);
        [t_perturbed,R_perturbed] = perturb(t,R,0.01);
        T_perturbed = [R_perturbed,t_perturbed];
        T_gt = [R,t];
        
        for a=1:num_algorithms
            Out = opengv(algorithms{a},indices{a},v1,v2,T_perturbed);
            [position_error, rotation_error] = evaluateTransformationError( T_gt, Out );
            position_errors(a,i) = position_error;
            rotation_errors(a,i) = rotation_error;
        end
        
        counter = counter + 1;
        if counter == 100
            counter = 0;
            display(['Iteration ' num2str(i) ' of ' num2str(iterations) '(noise level ' num2str(noise) ')']);
        end        
    end

    %Now compute the mean and median value of the error for each algorithm
    for a=1:num_algorithms
        mean_rotation_errors(a,n) = mean(rotation_errors(a,:));
        median_rotation_errors(a,n) = median(rotation_errors(a,:));
        mean_position_errors(a,n) = mean(position_errors(a,:));
        median_position_errors(a,n) = median(position_errors(a,:));
    end
    
end

%% Plot the results

figure(1)
plot(noise_levels,mean_rotation_errors,'LineWidth',2)
legend(names,'Location','NorthWest')
xlabel('noise level [pix]')
ylabel('mean rot. error [rad]')
grid on

figure(2)
plot(noise_levels,median_rotation_errors,'LineWidth',2)
legend(names,'Location','NorthWest')
xlabel('noise level [pix]')
ylabel('median rot. error [rad]')
grid on

figure(3)
plot(noise_levels,mean_position_errors,'LineWidth',2)
legend(names,'Location','NorthWest')
xlabel('noise level [pix]')
ylabel('mean pos. error [m]')
grid on

figure(4)
plot(noise_levels,median_position_errors,'LineWidth',2)
legend(names,'Location','NorthWest')
xlabel('noise level [pix]')
ylabel('median pos. error [m]')
grid on