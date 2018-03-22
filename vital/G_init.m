function [ net_G, opts ] = G_init()
%Initialize the asdn network

global gpu;

opts.useGpu = gpu;

% test policy
opts.batchSize_test = 256; % <- reduce it in case of out of gpu memory

opts.batchSize = 128;
opts.batch_pos = 32;

opts.learningRate_init = 0.001; % x10 for fc6

new_layers={};

%% init fc layers
scal = 1 ;
init_bias = 0.1;

last_dim=512;
% Block 4
new_layers{end+1} = struct('type', 'conv', ...
                           'name', 'fc1', ...
                           'filters', 0.01/scal * randn(3,3,last_dim,256,'single'),...
                           'biases', init_bias*ones(1,256,'single'), ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'filtersWeightDecay', 1, ...
                           'biasesWeightDecay', 0) ;
new_layers{end+1} = struct('type', 'relu', 'name', 'relu1') ;
%new_layers{end+1} = struct('type', 'dropout', 'name', 'drop1', 'rate', 0.5) ;

% Block 4
new_layers{end+1} = struct('type', 'conv', ...
                           'name', 'fc2', ...
                           'filters', 0.01/scal * randn(3,3,256,1,'single'),...
                           'biases', init_bias*ones(1,1,'single'), ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'filtersWeightDecay', 1, ...
                           'biasesWeightDecay', 0) ;

% new_layers{end+1}=struct('type','pdist', 'p', 2,...
%                         'noRoot','false','epsilon', 1e-6,...
%                         'aggregate','false', 'instanceWeights',[], 'name', 'l2_loss');              

losslayer.name = 'l2_loss';
losslayer.type = 'custom';
losslayer.forward = @forward;
losslayer.backward = @backward;
losslayer.class = [];

new_layers{end+1}=losslayer;
                    
net_G.layers=new_layers;

if opts.useGpu
    net_G = vl_simplenn_move(net_G, 'gpu') ;    
else
    net_G = vl_simplenn_move(net_G, 'cpu') ;    
end

function res_ = forward(layer, res, res_)
    res_.x = l2lossForward(res.x, layer.class);
end

function res = backward(layer, res, res_)
    res.dzdx = l2lossBackward(res.x, layer.class, res_.dzdx);
end

end



