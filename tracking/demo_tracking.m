clear all;
clc;

addpath('../utils');
addpath('../models');
addpath('../vital');

run ../matconvnet/matlab/vl_setupnn ;

global gpu;
gpu=true;
    
test_seq='Bolt';
conf = genConfig('otb',test_seq);

net=fullfile('../models/otbModel.mat');

result = vital_run(conf.imgList, conf.gt(1,:), net, true);


    


