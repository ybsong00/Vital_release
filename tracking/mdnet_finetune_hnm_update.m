function [net,net_asdn,poss,hardnegs] = mdnet_finetune_hnm_update(net,net_asdn,pos_data,neg_data,varargin)
% 
%
global gpu;

opts.useGpu = gpu;
opts.conserveMemory = true ;
opts.sync = true ;

opts.maxiter = 30;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;

opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
            class(net.layers{i}.filters)) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
            class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if opts.useGpu
            net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum);
            net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum);
        end
    end
end

for i=1:numel(net_asdn.layers)
    if strcmp(net_asdn.layers{i}.type,'conv')
        net_asdn.layers{i}.filtersMomentum = zeros(size(net_asdn.layers{i}.filters), ...
            class(net_asdn.layers{i}.filters)) ;
        net_asdn.layers{i}.biasesMomentum = zeros(size(net_asdn.layers{i}.biases), ...
            class(net_asdn.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if opts.useGpu
            net_asdn.layers{i}.filtersMomentum = gpuArray(net_asdn.layers{i}.filtersMomentum);
            net_asdn.layers{i}.biasesMomentum = gpuArray(net_asdn.layers{i}.biasesMomentum);
        end
    end
end

%% initilizing
if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
res = [] ;
res_asdn=[];
res_test=[];

n_pos = size(pos_data,4);
n_neg = size(neg_data,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;
lr_asdn = opts.learningRate*5;

% for saving positives
poss = [];

% for saving hard negatives
hardnegs = [];

% objective fuction
objective = zeros(1,opts.maxiter);

%% training on training set
% fprintf('\n');
for t=1:opts.maxiter
%     fprintf('\ttraining batch %3d of %3d ... ', t, opts.maxiter) ;
    iter_time = tic ;
    
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        batch = neg_data(:,:,:,...
            train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm));
        if opts.useGpu
            batch = gpuArray(batch) ;
        end
        
        % backprop
        net.layers{end}.class = ones(opts.batchSize_hnm,1,'single') ;
        res = vl_simplenn(net, batch, [], res, ...
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;        
        
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = ...
            squeeze(gather(res(end-1).x(1,1,2,:)));
    end
    [~,ord] = sort(score_hneg,'descend');
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data(:,:,:,hnegs);
%     fprintf('hnm: %d/%d, ', opts.batch_neg, opts.batchSize_hnm*opts.batchAcc_hnm) ;
    hardnegs = [hardnegs; hnegs];
    
    
%----------------------------yb added---------------------------    
    batch_asdn=pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos));
    if opts.useGpu
        batch_asdn = gpuArray(batch_asdn) ;
    end
    
    res_asdn=vl_simplenn(net_asdn,batch_asdn,[],res_asdn,...
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync);
    feat_asdn=squeeze(gather(res_asdn(end-1).x(:,:,1,:))); 
                    
    num=size(feat_asdn,3);
    mask_asdn=ones(3,3,512,num,'single');
    for i=1:num
        feat_=feat_asdn(:,:,i);
        featlist=reshape(feat_,9,1);
        [~,idlist]=sort(featlist,'ascend'); 
        idxlist=idlist(1:3);
        
        for k=1:length(idxlist)
            idx=idxlist(k);
            row=floor((idx-1)/3)+1;
            col=mod((idx-1),3)+1;
            mask_asdn(col,row,:,i)=0;
        end
    end
    batch_asdn=batch_asdn.*mask_asdn;    
%---------------------------------------------------------------         
    
    % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    poss = [poss; train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)];
    
    %batch = cat(4,pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        %im_hneg);
    batch = cat(4,batch_asdn,im_hneg);
    labels = [2*ones(opts.batch_pos,1,'single');ones(opts.batch_neg,1,'single')];
    if opts.useGpu
        batch = gpuArray(batch) ;
    end     
    
    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, batch, one, res, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync) ;
    
    % gradient step
    for l=1:numel(net.layers)
        if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
        
        net.layers{l}.filtersMomentum = ...
            opts.momentum * net.layers{l}.filtersMomentum ...
            - (lr * net.layers{l}.filtersLearningRate) * ...
            (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
            - (lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1} ;
        
        net.layers{l}.biasesMomentum = ...
            opts.momentum * net.layers{l}.biasesMomentum ...
            - (lr * net.layers{l}.biasesLearningRate) * ....
            (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
            - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;
        
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end
    
    % print information
    objective(t) = gather(res(end).x)/opts.batchSize ;
    iter_time = toc(iter_time);
    fprintf('objective %.3f, %.2f s\n', mean(objective(1:t)), iter_time) ;
    
%%-------------------yb added----------------------
    iter_time = tic;
    net_fc=net;
    net_fc.layers=net_fc.layers(1:end-1);
    prob_k=zeros(9,1);
    for i=1:9
        row=floor((i-1)/3)+1;
        col=mod((i-1),3)+1;
        batch=pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos));
        batch(col,row,:,:)=0;
        
        if opts.useGpu
            batch = gpuArray(batch) ;
        end 
        
        res_test = vl_simplenn(net_fc, batch, [], res_test, ...
            'disableDropout', true, ...
            'conserveMemory', true, ...
            'sync', true) ;
        
        feat = gather(res_test(end).x) ;
        
        X=feat;
        E = exp(bsxfun(@minus, X, max(X,[],3))) ;
        L = sum(E,3) ;
        Y = bsxfun(@rdivide, E, L) ;
        prob_k(i)=sum(Y(1,1,1,:));
    end
    [~,idx]=min(prob_k);
    row=floor((idx-1)/3)+1;
    col=mod((idx-1),3)+1;

    batch=pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos));
    labels = ones(3,3,1,opts.batch_pos,'single');
    labels(col,row,:)=0;
    
    if opts.useGpu
        batch = gpuArray(batch) ;
    end
        
    net_asdn.layers{end}.class = labels ;
    res_asdn = vl_simplenn(net_asdn, batch, one, res_asdn, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync) ;
    
    for l=1:numel(net_asdn.layers)
        if ~strcmp(net_asdn.layers{l}.type, 'conv'), continue ; end
        
        net_asdn.layers{l}.filtersMomentum = ...
            opts.momentum * net_asdn.layers{l}.filtersMomentum ...
            - (lr_asdn * net_asdn.layers{l}.filtersLearningRate) * ...
            (opts.weightDecay * net_asdn.layers{l}.filtersWeightDecay) * net_asdn.layers{l}.filters ...
            - (lr_asdn * net_asdn.layers{l}.filtersLearningRate) / opts.batchSize * res_asdn(l).dzdw{1} ;
        
        net_asdn.layers{l}.biasesMomentum = ...
            opts.momentum * net_asdn.layers{l}.biasesMomentum ...
            - (lr_asdn * net_asdn.layers{l}.biasesLearningRate) * ...
            (opts.weightDecay * net_asdn.layers{l}.biasesWeightDecay) * net_asdn.layers{l}.biases ...
            - (lr_asdn * net_asdn.layers{l}.biasesLearningRate) / opts.batchSize * res_asdn(l).dzdw{2} ;
        
        net_asdn.layers{l}.filters = net_asdn.layers{l}.filters + net_asdn.layers{l}.filtersMomentum ;
        net_asdn.layers{l}.biases = net_asdn.layers{l}.biases + net_asdn.layers{l}.biasesMomentum ;
    end
    objective(t) = gather(res_asdn(end).x)/opts.batchSize ;
    iter_time = toc(iter_time);
    fprintf('asdn objective %.3f, %.2f s\n', mean(objective(1:t)), iter_time) ;
    
%------------------------------------------------------------
end % next batch
