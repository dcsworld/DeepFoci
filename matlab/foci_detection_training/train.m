clc;clear all;close all force;
addpath('../utils')


rng(42)

data_path='../../../tmp/cell_lines';


data_chanels = {'a','b'};
matReaderData = @(x) matReader(x,'data',data_chanels,'norm_perc');
mask_chanels = {'a','b','ab'};
matReaderMask = @(x) matReader(x,'mask',mask_chanels,'norm_no');
model_name = 'individual_stainings';

folds = 5;

paralel_load = 1;


files = subdirx([data_path '/*data_53BP1.mat']);
in_layers = length(data_chanels);
out_layers = length(mask_chanels);



for fold = 1%:folds
    
    tmp_folder = ['../../../resutls_' model_name '_' num2str(fold)];
    mkdir(tmp_folder)
    

    [files_test,files_train_valid] = subfolder_based_split(files,fold,folds,42);
    files_test = files_test';


    train_valid_ind = randperm(length(files_train_valid));
    tmp = 1:round(length(files_train_valid)*0.75);
    
    train_ind = train_valid_ind(tmp);
    valid_ind = train_valid_ind;
    valid_ind(tmp) = [];
    
    files_valid = files_train_valid(valid_ind);
    files_train = files_train_valid(train_ind);
    
 
    
    

    volds = imageDatastore(data_path,'FileExtensions','.mat','IncludeSubfolders',1,'ReadFcn',matReaderData);
    volds = create_4_for_each(volds,files_train,data_path);

    volds_gt = imageDatastore(data_path,'FileExtensions','.mat','IncludeSubfolders',1,'ReadFcn',matReaderMask);
    volds_gt = create_4_for_each(volds_gt,files_train,data_path);

    volds_val = imageDatastore(data_path,'FileExtensions','.mat','IncludeSubfolders',1,'ReadFcn',matReaderData);
    volds_val = create_4_for_each(volds_val,files_valid,data_path);

    volds_gt_val = imageDatastore(data_path,'FileExtensions','.mat','IncludeSubfolders',1,'ReadFcn',matReaderMask);
    volds_gt_val = create_4_for_each(volds_gt_val,files_valid,data_path);


    
    patchSize = [96 96 48];
    patchPerImage = 1;
    miniBatchSize = 8;
    patchds = randomPatchExtractionDatastore(volds,volds_gt,patchSize,'PatchesPerImage',patchPerImage);
    patchds.MiniBatchSize = miniBatchSize;


    patchds_val = randomPatchExtractionDatastore(volds_val,volds_gt_val,patchSize,'PatchesPerImage',patchPerImage);
    patchds.MiniBatchSize = miniBatchSize;


    dsTrain = transform(patchds,@augment3dPatch);
    dsValid = transform(patchds,@augment3dPatch_valid); 
    
    
    disp('minibatchqueue train')
    mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'DispatchInBackground',paralel_load,...
    'MiniBatchFcn',@preprocessMiniBatch,...
    'MiniBatchFormat',{'SSSCB','SSSCB'});

    disp('minibatchqueue valid')
    mbq_val = minibatchqueue(dsValid,...
    'MiniBatchSize',miniBatchSize,...
    'DispatchInBackground',paralel_load,...
    'MiniBatchFcn',@preprocessMiniBatch,...
    'MiniBatchFormat',{'SSSCB','SSSCB'});

    
    lgraph = createUnet3d([patchSize in_layers],out_layers);
    dlnet = dlnetwork(lgraph);
    

    figure();
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    lineLossValid = animatedline('Color',[0 0.4470 0.7410]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on


    
    learnRate = 0.001;
    learnRateMult = 0.1;
    stepEpoch = [25 35 40];

    numEpochs = stepEpoch(end);
    
    gradDecay = 0.9;
    sqGradDecay = 0.999;
    epsilon = 1e-8;
    plot_train_freq = 40;
    valid_freq = round(2 * patchds.NumObservations/miniBatchSize);
    
    iteration = 0;
    start = tic;
    averageGrad = [];
    averageSqGrad = [];
    losses_train = [];
    

    grad_fcn = @modelGradients;

    disp('start training')
    
    % Loop over epochs.
    for epoch = 1:numEpochs
        
        if any(epoch == stepEpoch)
            learnRate = learnRate*learnRateMult;
        end
        
        % Shuffle data.
        shuffle(mbq);

        % Loop over mini-batches.

        while hasdata(mbq)
            
            iteration = iteration + 1;
            disp(iteration)
            
            % Read mini-batch of data.
            tic
            [dlX, dlY] = next(mbq);
            toc
            
            [gradients,state,loss] = dlfeval(grad_fcn,dlnet,dlX,dlY);
            dlnet.State = state;

            [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);
            
            loss = double(gather(extractdata(loss)));
            
            losses_train = [losses_train,loss];
            

            
            
            if mod(iteration,plot_train_freq) == 0
                
                
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                addpoints(lineLossTrain,iteration,mean(losses_train))
                title("Epoch: " + epoch + ", Elapsed: " + string(D) + '  , iteration per epoch: ' + round(patchds.NumObservations/miniBatchSize) )
                drawnow
                
                losses_train = [];
            end
            
            
            if mod(iteration,valid_freq) == 0 || iteration==1
                
                shuffle(mbq_val);

                % Loop over mini-batches.
                losses_valid = [];
                while hasdata(mbq_val)
                    
                    tic
                    [dlX, dlY] = next(mbq_val);
                    toc
                    
                    [dlYPred,state] = forward(dlnet,dlX);
                    loss = MSEpixelLoss(dlYPred,dlY);

                    loss = double(gather(extractdata(loss)));
                    
                    losses_valid = [losses_valid,loss]; 
                end
                
                
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                addpoints(lineLossValid,iteration,mean(losses_valid))
                title("Epoch: " + epoch + ", Elapsed: " + string(D) + '  , iteration per epoch: ' + round(patchds.NumObservations/miniBatchSize) )
                drawnow

                
            end
            
        end

    end

    save([tmp_folder '/final_net.mat'],'dlnet')
    save([tmp_folder '/names.mat'],'files_test','files_valid','files_train')
    
    print([tmp_folder '/train_curve'],'-dpng')

    
    

end