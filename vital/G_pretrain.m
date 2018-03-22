function [ net ] = G_pretrain( net_fc, net, pos_data,varargin )
%pretrain_asdn
%   

global gpu;
opts.useGpu = gpu;
opts.conserveMemory = true ;
opts.sync = true ;

opts.maxiter = 100;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.batchSize_hnm = 256;
opts.batchAcc_hnm = 4;

opts.batchSize = 128;
opts.batch_pos = 32;

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


%% initilizing
if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
res = [] ;

%%----------------Evaluate mask--------------------
n = size(pos_data,4);
nBatches = ceil(n/opts.batchSize);

net_fc.layers = net_fc.layers(1:end-1);

prob_k=zeros(9,1);
for k=1:9
    
row=floor((k-1)/3)+1;
col=mod((k-1),3)+1;
       
for i=1:nBatches
    
    batch = pos_data(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    batch(col,row,:,:)=0;
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = vl_simplenn(net_fc, batch, [], [], ...
        'disableDropout', true, ...
        'conserveMemory', true, ...
        'sync', true) ;
    
    f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;    
end

    X=feat;
    E = exp(bsxfun(@minus, X, max(X,[],3))) ;
    L = sum(E,3) ;
    Y = bsxfun(@rdivide, E, L) ;
    prob_k(k)=sum(Y(1,1,1,:));
end

[~,idx]=min(prob_k);
row=floor((idx-1)/3)+1;
col=mod((idx-1),3)+1;

%%-------------------------------------------------


% learning rate
lr = opts.learningRate;

% for saving positives
poss = [];

% objective fuction
objective = zeros(1,opts.maxiter);

n_pos = size(pos_data,4);
train_pos_cnt = 0;

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

res=[];
for t=1:opts.maxiter
    iter_time = tic ;
    
    poss = [poss; train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)];
    
    batch = pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos));
    labels = ones(3,3,1,opts.batch_pos,'single');    
    labels(col,row,:)=0;
    
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
            - (lr * net.layers{l}.biasesLearningRate) * ...
            (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
            - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;
        
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end
    objective(t) = gather(res(end).x)/opts.batchSize ;
    iter_time = toc(iter_time);
    fprintf('asdn objective %.3f, %.2f s\n', mean(objective(1:t)), iter_time) ;
end


end

