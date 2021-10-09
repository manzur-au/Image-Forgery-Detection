% Author: Mohammad Manzurul Islam
% Code created: 20/11/2018
% Last Modified: 11/5/2019
% A Program for Extracting features from image - a part of code for implementing the paper:
% A Robust Forgery Detection Method for Copy–Move and Splicing Attacks in Images
% https://doi.org/10.3390/electronics9091500

%%
clear
tic;


%% Select your dataset

%CASIA2 image dataset
% imageFiles = [dir('D:/IoTDataset/CASIA/CASIA2/*/*.jpg');dir('D:/IoTDataset/CASIA/CASIA2/*/*.tif');dir('D:/IoTDataset/CASIA/CASIA2/*/*.bmp')];
% imgDataset = 'CASIA2Gray';

% FBDDF - 400 TIF
% imageFiles = dir('C:/Users/mmislam/OneDrive - Federation University Australia/2ndYear/2ndYear/workForJournal/FBDDF-TIF/400_TrainTestSet_TIF/*.tif');
% imageFiles = [dir('./400_TrainTestSet_TIF/*.jpg');dir('./400_TrainTestSet_TIF/*.tif')];
%imageFiles = [dir('../200_original/*.jpg');dir('../200_forged_tif/tif_Uncompressed_no_layer/*.tif')];

% % FBDDF PS12 Rotated
imageFiles = [dir('./FBDDF_PS12_random_rotated/*.jpg')];

% FBDDF PS5 Scaling
% imageFiles = [dir('./FBDDF_PS5_scaling_90/*.jpg')];


% FBDDF - 400 JPEG-5
% imageFiles = [dir('../200_original/*.jpg');dir('../200_forged_jpeg/jpeg_Compression_PS_5/*.jpg')];
imgDataset = 'FBDDF_Proposed_PS12_Random_Rotated_Gray';

%% Extract features

numberOfFiles = length(imageFiles);
for (bsizes=[4,8,16])
    blockSize =bsizes;
    dataPerBlock = blockSize^2;
    
    D = dctmtx(blockSize);
    [r,c] = ndgrid(0:blockSize-1);
    [~,idZigZag] = sortrows([r(:)+c(:),(r(:)-c(:)).*(2*(mod(r(:)+c(:),2)) - 1)],[1,2]);
    funDCT = @(block_struct) D*block_struct.data*D';
    funZigZag = @(block_struct) block_struct.data(idZigZag);
    fAVG = nan(numberOfFiles,dataPerBlock+1);
    %%
    %Read all the image in a folder
    for imgNo = 1:numberOfFiles
        folderName = imageFiles(imgNo).folder;
        imgPath = [folderName '\' imageFiles(imgNo).name];
        img = im2double(imread(imgPath));
        
        if ndims(img) == 3 %Check if image is in color or gray scale.
            imgGray = rgb2gray(img);
        else
            imgGray = img;
        end
        
        %% Calculate Block DCT
        
        DCT = zeros(size(imgGray));
        DCT = blockproc(imgGray,[blockSize blockSize],funDCT,'PadPartialBlocks',true,'PadMethod',0.5);
        %Claculate the magnitutde component of DCT
        imDCT = abs(DCT);
        %% This section Calculates LBP of the entire 2D - DCT array
        imgGrayLBP = imDCT;
        
        [rows columns numberOfColorBands] = size(imgGrayLBP);
        LBP = zeros(size(imgGrayLBP));
        
        for row = 2 : rows - 1
            for col = 2 : columns - 1
                centerPixel = imgGrayLBP(row, col);
                pixel7=imgGrayLBP(row-1, col-1) >= centerPixel;
                pixel6=imgGrayLBP(row-1, col) >= centerPixel;
                pixel5=imgGrayLBP(row-1, col+1) >= centerPixel;
                pixel4=imgGrayLBP(row, col+1) >= centerPixel;
                pixel3=imgGrayLBP(row+1, col+1) >= centerPixel;
                pixel2=imgGrayLBP(row+1, col) >= centerPixel;
                pixel1=imgGrayLBP(row+1, col-1) >= centerPixel;
                pixel0=imgGrayLBP(row, col-1) >= centerPixel;
                eightBitNumber = uint8(...
                    pixel7 * 2^7 + pixel6 * 2^6 + ...
                    pixel5 * 2^5 + pixel4 * 2^4 + ...
                    pixel3 * 2^3 + pixel2 * 2^2 + ...
                    pixel1 * 2 + pixel0);
                LBP(row, col) = eightBitNumber;
            end
        end
        
        
        %%
        %Block Processing for getting the zigzag pattern of the localBinaryPatternImage matrix.
        
        LBP_ZigZagTemp1 = blockproc(LBP,[blockSize blockSize],funZigZag,'PadPartialBlocks', true);
        
        LBP_ZigZagTemp2=LBP_ZigZagTemp1';
        
        
        %Now we want to make zigzag pattern of each block into separate rows as     
        [rowzz colzz] = size(LBP_ZigZagTemp2);
        i = 1;
        k = 0;
        tok = 1;
        
        while i<=rowzz
            j = 1;
            while j<=colzz
                k = k+1;
                LBP_ZigZag(tok, k) = LBP_ZigZagTemp2(i,j);
                if mod(j,dataPerBlock) == 0
                    tok = tok+1;
                    k = 0;
                end
                j = j+1;
            end
            i = i+1;
        end
        
        avg = mean(LBP_ZigZag);
        fAVG(imgNo,1:dataPerBlock) = avg;
        %end
        
        
        
        %% %% Now put the value of authentic(0) and spliced image(1). Check REGEX based on your dataset
        
        
        if regexp(imageFiles(imgNo).name,'\w*(_org.jpg)')
            fAVG(imgNo,dataPerBlock+1) = 0;
        else
            fAVG(imgNo,dataPerBlock+1) = 1;
        end
        
        
    end %End of image in folders - FOR loop

    data = fAVG;
    dataset = sprintf('%s_%dx%d.mat',imgDataset,blockSize,blockSize);
    save(dataset, 'data');
    
end % end of block control loop
toc;
%% Additional helper codes
%Merge 3 blocks

% dataset = 'BDD_10k_FBDDF_400';
% data4_ = sprintf('%s_%dx%d.mat',dataset,4,4);
% data4=load(data4_);
% data8_ = sprintf('%s_%dx%d.mat',dataset,8,8);
% data8=load(data8_);
% data16_ = sprintf('%s_%dx%d.mat',dataset,16,16);
% data16=load(data16_);
% data = [data4.data(:,1:16) data8.data(:,1:64) data16.data(:,:)];
% allBlocks=sprintf('%s_4x8x16.mat',dataset);
% save(allBlocks, 'data');
