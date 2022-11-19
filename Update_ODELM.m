function [K,U,nLabeling,b_k,indexLabeling,MS,u] = Update_ODELM(K,U,H_test,label,yPred,Ct,nLabeling,u,b_k,Bw,iter,update_strategy)

% Label:One-Hot
A = zeros(6,1);
A(label) = 1;
label = A;
clear A

% Query function
y = sort(yPred);
MS = y(end)-y(end-1);
if MS < u 
    labeling = 1;  
    u = u * 1.0001;  % s = 0.0001
else
    labeling = 0;
    u = u * 0.9999;
end

% update ODELM
if b_k == Bw
    labeling = 0;
end  
H_test = H_test';
label = label';
if labeling
    switch update_strategy
        case 'FW'
            % The ODELM with fixed weight
            K = K + Ct*(H_test')*H_test;
            U = U + Ct*(H_test') *label;
        case 'LinW'
            % The ODELM with linear-growth weight
            K = K + (nLabeling+1)*Ct*(H_test')*H_test;  
            U = U + (nLabeling+1)*Ct*(H_test') *label;
        case 'LogW'
            % The ODELM with logarithmic-growth weight
            K = K + log(nLabeling+1)*Ct*(H_test')*H_test;
            U = U + log(nLabeling+1)*Ct*(H_test') *label;
    end
end

nLabeling = nLabeling + labeling;   
b_k = b_k + labeling;                
if labeling == 1
    indexLabeling = iter;
else
    indexLabeling = [];
end