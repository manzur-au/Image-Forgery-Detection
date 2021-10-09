% Author: Mohammad Manzurul Islam
% Code created: 20/11/2018
% Last Modified: 11/5/2019
% A Program for combining blocks - a part of code for implementing the paper:
% A Robust Forgery Detection Method for Copy–Move and Splicing Attacks in Images
% https://doi.org/10.3390/electronics9091500

dataset = 'YOUR DATASET NAME';
for(bsizes = [4,8,16])   
    blockSize = bsizes;
    dataset_ = sprintf('%s_%dx%d.mat',dataset,blockSize,blockSize);
    v = strcat('data',num2str(bsizes));
    blockData.(v) = load(dataset_);
end

data = [blockData.data4.data(:,1:end-1) blockData.data8.data(:,1:end-1) blockData.data16.data];

allBlocks=sprintf('%s_4x8x16.mat',dataset);
save(allBlocks, 'data');

%% CombineCbCr
% clear
% load('FBDDF_Proposed_TIF_Uncomp_400_TMM_CbCr_4x8x16.mat');
% Cb = data(:,1:end-1);
% load('FBDDF_Proposed_TIF_Uncomp_400_TMM_CbCr_4x8x16.mat')
% Cr = data;
% data = [Cb Cr];
% save('FBDDF_Proposed_TIF_Uncomp_400_TMM_CbCr_4x8x16-1.mat', 'data');
%% Combine blocks
% clear
% load('CASIA2_Cr_4x4.mat');
% var4 = data(:,1:end-1);
% load('CASIA2_Cr_8x8.mat');
% var8 = data(:,1:end-1);
% load('CASIA2_Cr_16x16.mat');
% var16 = data;
% clear data;
% data = [var4 var8 var16];
% save('CASIA2_Cr_4x8x16.mat', 'data');