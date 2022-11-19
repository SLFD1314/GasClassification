function [Acc_Train,K,U] = Training_ODELM(Training_Source,Training_TransferSample, nHidden, AF,Cs,Ct)
s = RandStream('dsfmt19937');
% Source domain data
Sourcedata = Training_Source;
Source_Label = Sourcedata(:,1)';                       
Source_Instance = Sourcedata(:,2:size(Sourcedata,2))'; 

% Transfer Sample set
Transferdata = Training_TransferSample;
Transfer_Label = Transferdata(:,1)';                          
Transfer_Instance = Transferdata(:,2:size(Transferdata,2))';  

% Number of Samples and Number of Input Layer Neurons
nSource = size(Source_Instance,2);                  
nTransfer = size(Transfer_Instance,2);               
nInput = size(Source_Instance,1);                   

% Label:One-Hot
if max(Source_Label)<6
    I = eye(6);
else
    I = eye(max(Source_Label));
end
Source_Label = (I(Source_Label,:))';         
if max(Transfer_Label)<6
    It = eye(6);
else
    It = eye(max(Transfer_Label));
end
Transfer_Label = (It(Transfer_Label,:))';     
                                  
% Calculate the weights and bias from the input layer to the hidden layer
IW = rand(s,nHidden,nInput)*2 - 1;            
Bias = rand(s,nHidden,1);                     
tempH = IW * Source_Instance;                 
tempHt = IW * Transfer_Instance;             
ind = ones(1,nSource);           
indt = ones(1,nTransfer);      
BiasMatrix = Bias(:,ind);        
BiasMatrixT = Bias(:,indt);     

% Calculate the output from the input layer to the hidden layer
tempH = tempH + BiasMatrix;                 
tempHt = tempHt + BiasMatrixT;            
H = ActivationFunction(tempH,AF);
Ht = ActivationFunction(tempHt,AF);
clear tempH;                                    
clear tempHt;

% Calculate the weight from hidden layer to output layer
H = H';                                              
Ht = Ht';                                            
Source_Label = Source_Label';                        
Transfer_Label = Transfer_Label';                    
K = speye(nHidden) + Cs*(H')*H+Ct*(Ht')*Ht;
U = Cs*H'*Source_Label + Ct*Ht'*Transfer_Label;
OutputWeight = inv(K) * U;

% Predict the lables of training set  
Predict_Source = (H * OutputWeight)';              
Predict_Transfer = (Ht * OutputWeight)';           
clear H;
clear Ht;

% Calculate accuracy of training set
Source_Label = Source_Label';                       
Transfer_Label = Transfer_Label';                  
[~,result_source] = max(Predict_Source);
[~,result_transfer] = max(Predict_Transfer);
Acc_Train = roundn((sum((result_source')==Sourcedata(:,1))+sum((result_transfer')==Transferdata(:,1)))...
            /(size(Source_Label,2)+size(Transfer_Label,2))*100,-2);

% Activation function  
function H = ActivationFunction(temp,AF)
switch lower(AF)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-temp));
    case {'sin','sine'}
        H = sin(temp);
    case {'hardlim'}
        H = double(hardlim(temp));
    case {'tribas'}
        H = tribas(temp);
    case {'radbas'}
        H = radbas(temp);
    case 'tanh'
        H = 1-2./(exp(2*temp)+1);
    case 'logsig'
        H =1./(1+exp(-temp));
    case 'linear'
        H = temp;
    case {'relu'}
        temp(temp<0) = 0;
        H = temp;
end