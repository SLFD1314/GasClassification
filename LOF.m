function Local_Outlier_Factor = LOF(XS,k)
% LOF
% k is the number of k_neighborhood
% XS feature set 
Dis_Euclidean  = squareform(pdist(XS));
[dis,sample_index] = sort(Dis_Euclidean);
k_distance = dis(2:k+1,:);
k_neighborhood = sample_index(2:k+1,:);
local_reach_distance = zeros(size(XS,1),1);
for i = 1:size(XS,1)
    reach_distance = max([k_distance(:,i) (k_distance(k,k_neighborhood(:,i)))'],[],2);
    local_reach_distance(i) = mean(reach_distance)+1e-10;
end
Local_Outlier_Factor = zeros(size(XS,1),1);
for i = 1:size(XS,1)
    Local_Outlier_Factor(i) = mean(local_reach_distance(k_neighborhood(:,i)))/local_reach_distance(i);
end



