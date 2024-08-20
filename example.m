% Example of using emd.m for separation and diarization
clear
close all

% loads auxilliary functions (STFT, KL_NMF, etc.) 
addpath aux_tools/

J = 3;              % number of sources
cPerSrc = 20;       % components of NMF assigned to each source
stft_win_len = 512; % length of the STFT analysis window
maxNumIter = 10;    % number of EM iterations
snr = 10;           % level of corruption of param's initialization (dB)
%snr = Inf;         % this is the perfect initialization (no corruption)

% {J} x [M x I] load the J true sources
[y,fs] = arrayfun(@(j) audioread(sprintf('trueSrc%d.wav',j)), 1:J, 'uniformoutput', false);

% [M x I x J] array with ground-truth sources
y = cat(3,y{:});

% [M x I] mixture signal
x = sum(y,3);

% write mixture signal
audiowrite('mix.wav', x/max(abs(x(:))) ,fs{1});

% initialization of NMF parameters (using one of the microphones)
[W,H,Kj] = corruptInit( y , cPerSrc, stft_win_len , snr);

fprintf('Separating & Diarizing ..\n');

% emd.m implements the EM algorithm
ye = emd(x,W,H,Kj,maxNumIter,stft_win_len,fs{1});

% write separated sources
arrayfun(@(j) audiowrite(sprintf('estimatedSrc%d.wav',j), ye(:,:,j) / max(max(abs(ye(:,:,j)))) ,fs{1}), 1:J , 'uniformoutput',false)

















