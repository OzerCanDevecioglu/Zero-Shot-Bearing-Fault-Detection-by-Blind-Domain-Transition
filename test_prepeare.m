load 'temats/noisy_sig.mat'
load 'temats2/clear_sig.mat'

noisy_sig2=noisy_sig;
clear_sig2=clear_sig;

load 'temats/noisy_sig.mat'
load 'temats/clear_sig.mat'


    id=10;
    fname=[ 'goutputs/ch1data_train_' int2str(id) '.mat' ];
    load(fname)


ns1=noisy_sig;
ns2=noisy_sig2;


ch1array=[predicted; ns1];
labelarray=ones(size(ch1array,1),2);
labelarray(1:size(predicted,1),1)=labelarray(1:size(predicted,1),1)*-1;
labelarray((size(predicted,1)+1):(size(predicted,1)+size(ns1,1)),2) =labelarray((size(predicted,1)+1):size(predicted,1)+size(ns1,1),2)*-1;

i=1
save(['test_data/ch1data_train_' int2str(i) '.mat'],'ch1array')
save(['test_data/labels_train_' int2str(i) '.mat'],'labelarray')



cc=[clear_sig;clear_sig2];

ch1array=[cc; ns2];
labelarray=ones(size(ch1array,1),2);
labelarray(1:size(cc,1),1)=labelarray(1:size(cc,1),1)*-1;
labelarray((size(cc,1)+1):(size(cc,1)+size(ns2,1)),2) =labelarray((size(cc,1)+1):size(cc,1)+size(ns2,1),2)*-1;




save(['test_data/ch1data_test_' int2str(i) '.mat'],'ch1array')
save(['test_data/labels_test_' int2str(i) '.mat'],'labelarray')
