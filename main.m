clear all ;

%% Parameters 

% CSA = -1;
% NOR = 0;
% OSA = 1;
% HYP = 2;
% MSA = 3;
% 
% CurrentWindowAR =     10;  % 10-seconds
% CurrentWindowARlen =  20;  % 20 windows
% CurrentWindowFR =     10;  % 10-seconds
% CurrentWindowFRlen =  20;  % 20 windows
% CurrentWindowCov =    10;  % 10-seconds
% CurrentWindowCovlen = 20;  % 20 windows
% 
% PreviousWindowAR =    60;   % 60-seconds
% PreviousWindowARlen = 120;  % 120 windows
% 
% doSST = 0;
% 
% fs = 100;
% fsDown = 4;
% downRate = 25;
% 
% noWAKE = 0;
% 
% SLIDE = 2;  % 1/SLIDE-second range for one window
% 
% QtAR = 0.95; 

LoadParameters;

%% Input Test Data

addpath('database/testingdata')

pnum = csvread('pnum47.csv',0,0);

pnum = pnum(randsample(47, 5));

[~, ~, alldata] = xlsread('LightOff1.xlsx');

Channels = cell(length(pnum), 3);
state = cell(length(pnum), 1);
    
for i = 1 : length(pnum)
    
    fprintf(['*** Input data with patient number ', num2str(pnum(i)), '\n']);
    
    tar = find(cell2mat(alldata(:, 1)) == pnum(i));
    
    Time = datetime(alldata{tar, 2}, 'ConvertFrom', 'excel');
    
    Ht = hour(Time);      if Ht >= 21;  Ht = Ht-12;  end
    Mt = minute(Time);
    St = second(Time);
    
    format long
    t_lightoff = datenum([num2str(Ht) ':' num2str(Mt) ':' num2str(St)],'HH:MM:SS');
    
    Time = erase(alldata{tar,3},"¤U¤È");
    
    Ht = hour(Time);      if Ht >= 21;  Ht = Ht-12;  end
    Mt = minute(Time);
    St = second(Time);
    
    format long
    t_StartRecord = datenum([num2str(Ht) ':' num2str(Mt) ':' num2str(St)],'HH:MM:SS');
    
    RecordOffSet = (t_lightoff-t_StartRecord)*24*60*60;      % offset in second
    fprintf(['****** Record offset is ',num2str(RecordOffSet/60),' mins\n']);
    
    [THO, ABD, T_LEN] = InputTestData_PSG(num2str(pnum(i)));
    
    state{i} = InputTestData_Event(num2str(pnum(i)), t_lightoff, T_LEN);
    
    Channels{i, 1} = THO;
    Channels{i, 2} = ABD;
    Channels{i, 3} = zeros(T_LEN/100, 1);  % Prepare for STAGE record
    
end

clear alldata THO ABD T_LEN Time Ht Mt St t_lightoff t_StartRecord

%% Training : Feature extraction 

Features = cell(length(pnum), 1);

sp = PreviousWindowARlen + 1;

RightBoundary = max([CurrentWindowFRlen, CurrentWindowCovlen, CurrentWindowARlen]) ;

ep = zeros(length(pnum), 1);

for i = 1 : length(pnum)
    
    ep(i) = min(state{i}.len, SLIDE*length(Channels{i, 1})/fsDown) - RightBoundary + 1;
    
    THO = Channels{i, 1};
    ABD = Channels{i, 2};
    STAGE = Channels{i, 3};
    
    sta = state{i};
    
    fprintf(['\nStart extract Features of ', num2str(pnum(i)), ' (No.', num2str(i), ')\n']);
    
    [AR_THO, AR_ABD] = Feature_AR(STAGE, sta.psg, THO, ABD, sp, ep(i));
    [FR_THO, FR_ABD] = Feature_FR(STAGE, sta.psg, THO, ABD, sp, ep(i));
    Cov_THOABD       = Feature_Cov(STAGE, sta.psg, THO, ABD, sp, ep(i));

    Features{i, 1} = [AR_THO, AR_ABD, FR_THO, FR_ABD];
    Features{i, 2} = Cov_THOABD;
    
end

clear THO ABD STAGE sta AR_THO AR_ABD FR_THO FR_ABD Cov_THOABD

%% 

y = randsample(length(pnum), 4);

Trainpnum = false(length(pnum), 1);
Trainpnum(y) = 1;
Trainpos = find(Trainpnum);

Testpnum = ~Trainpnum;
Testpos = find(Testpnum);

%% Training : Convert

ConvertInfo = cell(length(Trainpos), 1);

for i = 1 : length(Trainpos)
    
    sta = state{Trainpos(i)};
    
    ConvertInfo{i} = Convert(sta.psg, ep(Trainpos(i)));
    
end

CInfo = cell2mat(ConvertInfo);
F = cell2mat(Features(Trainpos, 1));

[TraTarget, TraTargetNum] = Organize(F, CInfo);
    
clear STAGE sta

%% Training : SVM model
    
fprintf('\nStart generate SVM structure.\n');
    
svmStruct_N_OC = SVM_N_OC(TraTarget, TraTargetNum);
svmStruct_O_C = SVM_O_C(TraTarget, TraTargetNum);

%% state machine

numTap = 12;

predictPSG = cell(length(Testpos), 1);

for i = 1 : length(Testpos)
    
    fprintf('\nStart state machine\n')
    
    THO = Channels{Testpos(i), 1};
    ABD = Channels{Testpos(i), 2};
    STAGE = Channels{Testpos(i), 3};
    
    sta = zeros(length(state{Testpos(i)}.psg), 1);
    LEN = length(sta);
    
    [FR_THO, FR_ABD] = Feature_FR(STAGE, sta, THO, ABD, sp, ep(Testpos(i)));
    Cov_THOABD       = Feature_Cov(STAGE, sta, THO, ABD, sp, ep(Testpos(i)));
    
    saF = 0;
    tF = 0;
    
    AR_THO = zeros(numTap, 1);
    AR_ABD = zeros(numTap, 1);
    
    for j = 0 : numTap - 2
        
        previousWinidx = fsDown*(sp+j-PreviousWindowARlen-1)/SLIDE+1 : fsDown*(sp+j-1)/SLIDE;
        currentWinidx = fsDown*(sp+j-1)/SLIDE+1 : fsDown*((sp+j-1)+CurrentWindowARlen)/SLIDE ;
        
        curTHO = THO(currentWinidx);
        curABD = ABD(currentWinidx);
        
        preTHO = THO(previousWinidx);
        preABD = ABD(previousWinidx);
        
        preTHO = abs(hilbert(preTHO-mean(preTHO))) + mean(preTHO);
        preABD = abs(hilbert(preABD-mean(preABD))) + mean(preABD);

        curTHO = abs(hilbert(curTHO-mean(curTHO))) + mean(curTHO);
        curABD = abs(hilbert(curABD-mean(curABD))) + mean(curABD);

        AR_THO(j+1) = log10( quantile(abs((curTHO)), QtAR) / quantile(abs((preTHO)), QtAR) );
        AR_ABD(j+1) = log10( quantile(abs((curABD)), QtAR) / quantile(abs((preABD)), QtAR) );
        
    end
    
    tic
    
    for ss = sp + numTap - 1 : ep(Testpos(i))
        
        curidx = ss-numTap+1:ss;
        
        currentWinidx = fsDown*(ss-1)/SLIDE + 1 : fsDown*((ss-1)+CurrentWindowARlen)/SLIDE;
        previousWinidx = fsDown*(ss-PreviousWindowARlen-1)/SLIDE+1 : fsDown*(ss-1)/SLIDE;
        
        curTHO = THO(currentWinidx);
        curABD = ABD(currentWinidx);
        
        preTHO = THO(previousWinidx);
        preABD = ABD(previousWinidx);
        
        preTHO = abs(hilbert(preTHO-mean(preTHO))) + mean(preTHO);
        preABD = abs(hilbert(preABD-mean(preABD))) + mean(preABD);

        curTHO = abs(hilbert(curTHO-mean(curTHO))) + mean(curTHO);
        curABD = abs(hilbert(curABD-mean(curABD))) + mean(curABD);
        
        if saF == 1
            
            standingWinidx = fsDown*((ss-1)-PreviousWindowARlen-1)/SLIDE+1 : fsDown*((ss-1)-1)/SLIDE;
            
            stdTHO = THO(standingWinidx);
            stdABD = ABD(standingWinidx);
            
            stdTHO = abs(hilbert(stdTHO-mean(stdTHO))) + mean(stdTHO);
            stdABD = abs(hilbert(stdABD-mean(stdABD))) + mean(stdABD);
        
        end
        
        if sta(ss-1) == NOR
            
            AR_THO(numTap) = log10( quantile(abs((curTHO)), QtAR) / quantile(abs((preTHO)), QtAR) );
            AR_ABD(numTap) = log10( quantile(abs((curABD)), QtAR) / quantile(abs((preABD)), QtAR) );
            
        else
            
            AR_THO(numTap) = quantile(abs((curTHO)), QtAR) / quantile(abs((stdTHO)), QtAR);
            AR_ABD(numTap) = quantile(abs((curABD)), QtAR) / quantile(abs((stdABD)), QtAR);
            
        end
        
        feature = [AR_THO, AR_ABD, FR_THO(curidx), FR_ABD(curidx)];
        
        [sta(ss), tF] = StateMachine(feature, Cov_THOABD(curidx), svmStruct_N_OC, svmStruct_O_C, sta(ss-1), tF);
        
        if sta(ss-1) == NOR && sta(ss) ~= NOR
            
            saF = 1;
            
        else 
            
            saF = 0;
            
        end
        
        AR_THO(1:numTap-1) = AR_THO(2:numTap);
        AR_ABD(1:numTap-1) = AR_ABD(2:numTap);
        
    end
    
    toc
    
    predictPSG{i} = sta;
    
end

%% Confusion matrix

cm = cell(length(Testpos), 1);

for i = 1 : length(Testpos)
    
    outputs = predictPSG{i};
    O = zeros(size(outputs));
    
    O(outputs == 0) = 1;    % NOR
    O(outputs == -1) = 2;   % CSA
    O(outputs == 1) = 3;    % OSA
    
    targets = state{Testpos(i)}.psg;
    T = zeros(size(targets));
    
    T(targets == 0) = 1;    % NOR
    T(targets == -1) = 2;   % CSA
    T(targets == 1) = 3;    % OSA
    T(targets == 2) = 2;    % HYP
    T(targets == 3) = 3;    % MSA
    
    Conf = zeros(3);
    
    for j = 1 : length(outputs)
        
        Conf(T(j), O(j)) = Conf(T(j), O(j)) + 1;
        
    end
    
    cm{i} = Conf;
    
end

%%

ClearParameters;
