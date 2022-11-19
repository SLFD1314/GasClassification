%% Gas classification task with sensor drift
% Sensor drift dataset: batch1-batch10. http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
clear
clc
%% Import training set and test set
% Training set: Batch 1
dataname = ['batch',num2str(1),'.dat'];
[Train_Label,Traning_Instance] = libsvmread(dataname);              % sparse double
Traning_Instance = full(Traning_Instance);                          % sparse double to full double
[Traning_Instance,ps] = mapminmax(Traning_Instance',-1,1);          
Traning_Instance = Traning_Instance';

% The source domain sample and the transfer sample are selected from training data
num_neighborhood = 20;  
Feature_LOF = LOF(Traning_Instance,num_neighborhood);   % Calculate LOF
nTransfer = 30;                                   % Number of transfer sample
nSource = 100;
[~,index_transfer] = sort(Feature_LOF);                 

% The source domain sample and the transfer sample are labeled
Source_Instance = Traning_Instance(index_transfer(1:nSource),:);                      
Source_Label = Train_Label(index_transfer(1:nSource));
Transfer_Instance = Traning_Instance(index_transfer(end-nTransfer+1:end),:);                   
Transfer_Label = Train_Label(index_transfer(end-nTransfer+1:end));

% Test set: Batch2 - Batch10
Test_Label = [];
Test_Instance = [];
for i = 2:10
    dataname = ['batch',num2str(i),'.dat'];
    [Label,Instance] = libsvmread(dataname);
    Instance = full(Instance);
    Test_Label = [Test_Label;Label];
    Test_Instance = [Test_Instance; Instance];
end
Test_Instance = mapminmax('apply',Test_Instance',ps); 
Test_Instance = Test_Instance';
%% Training ODELM
nHidden = 50;                             % Number of hidden layer nodes
AF = 'relu';                              % Activation function: relu linear sigmoid tanh logsig,etc
Cs = 0.10;                                % Penalty coefficient
Ct = 0.01;                                % Penalty coefficient
Training_Source = [Source_Label,Source_Instance];
Training_TransferSample = [Transfer_Label,Transfer_Instance];
[~,K,U] = Training_ODELM(Training_Source,Training_TransferSample, nHidden, AF,Cs,Ct); % Results of training
%% Online prediction and model update
acc_Test = zeros(10,1);
acc_batch = zeros(10,9);
bT_batch = zeros(10,9);
up_strategy = 'LinW';                   % The three different weight updating strategies: 'ODELM-FW','ODELM-LogW','ODELM-LinW'
for Bw = 1:10                           % The budget
    th1 = 0.1;                           
    K_test = K;
    U_test = U;    
    Total_yPred = zeros(6,length(Test_Label));
    Total_label = zeros(length(Test_Label),1);
    labeling_i = [];
    nLabeling = 0;                      % The cost of labeling the test set (target domain)
    w = 1;                              
    b_k = 0;                             

    for i = 1:length(Test_Label)
        Label_test = Test_Label(i);
        X_test = Test_Instance(i,:);
        TestingData = [Label_test,X_test];
        if w == 101
            w = 1;
            b_k = 0;
        end
        [yPred,H_test] = Testing_ODELM(K_test,U_test,TestingData, nHidden,AF);
        Total_yPred(:,i) = yPred;
        [~,pre_label] = max(yPred);
        Total_label(i) = pre_label;
        [K_test,U_test,nLabeling,b_k,indexLabeling,~,th1] = Update_ODELM(K_test,U_test,H_test,Label_test,yPred,Ct,nLabeling,th1,b_k,Bw,i,up_strategy);
        labeling_i = [labeling_i;indexLabeling];
        w = w + 1;
    end
    
    acc_Test(Bw) = roundn(sum(Total_label==Test_Label)/length(Test_Label)*100,-2);
    batch = [0 1244  2830  2991  3188  5488  9101  9395   9865 13465];
    
    for i =1:9
        acc_batch(Bw,i) = roundn(sum(Total_label(batch(i)+1:batch(i+1)) == Test_Label(batch(i)+1:batch(i+1)))/length(Total_label(batch(i)+1:batch(i+1)))*100,-2);
        bT_batch(Bw,i) = sum((labeling_i<=batch(i+1))&(labeling_i>=batch(i)+1));
    end
    disp(['************************************','   ODELM-',up_strategy,'   *******************************************'])
   
    disp(['Bw = ' num2str(Bw), ', Acc_Test = ' num2str(acc_Test(Bw)),', b = ' num2str(nLabeling+nTransfer+nSource)])
    
    disp(['Acc_Batch2 = ' num2str(acc_batch(Bw,1)),', Acc_Batch3 = ' num2str(acc_batch(Bw,2)),', Acc_Batch4 = ' num2str(acc_batch(Bw,3)),', Acc_Batch5 = ' num2str(acc_batch(Bw,4)),', Acc_Batch6 = ' num2str(acc_batch(Bw,5)),...
        ', Acc_Batch7 = ' num2str(acc_batch(Bw,6)),', Acc_Batch8 = ' num2str(acc_batch(Bw,7)),', Acc_Batch9 = ' num2str(acc_batch(Bw,8)),', Acc_Batch10 = ' num2str(acc_batch(Bw,9))]);
   
%     disp(['bT_Batch2 = ' num2str(bT_batch(Bw,1)),', bT_Batch3 = ' num2str(bT_batch(Bw,2)),', bT_Batch4 = ' num2str(bT_batch(Bw,3)),', bT_Batch5 = ' num2str(bT_batch(Bw,4)),', bT_Batch6 = ' num2str(bT_batch(Bw,5)),...
%         ', bT_Batch7 = ' num2str(bT_batch(Bw,6)),', bT_Batch8 = ' num2str(bT_batch(Bw,7)),', bT_Batch9 = ' num2str(bT_batch(Bw,8)),', bT_Batch10 = ' num2str(bT_batch(Bw,9))]);
end


