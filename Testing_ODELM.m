function [yPred,H_test] = Testing_ODELM(K,U,TestingData,nHidden,AF)
% test set
test_data =  TestingData;
TV.T = test_data(:,1)';                   
TV.P = test_data(:,2:size(test_data,2))'; 
clear test_data;                                    

% Number of Samples and Number of Input Layer Neurons
nTest =size(TV.P,2);                     
nInput = size(TV.P,1);

% Label:One-Hot
A = zeros(6,1);
A(TV.T) = 1;
TV.T = A;
clear A
                                              
% Calculate the weights and bias from the input layer to the hidden layer
s = RandStream('dsfmt19937');
IW = rand(s,nHidden,nInput)*2-1;           
Bias = rand(s,nHidden,1);                 

% Calculate the weight from hidden layer to output layer
OutputWeight = inv(K) * U;

% Predict result
tempH_test = IW * TV.P;                      
clear TV.P;                                        
ind = ones(1,nTest);                        
BiasMatrix = Bias(:,ind);                    
tempH_test = tempH_test + BiasMatrix;        
H_test = ActivationFunction(tempH_test,AF);
yPred = (H_test' * OutputWeight)';              

end

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
end