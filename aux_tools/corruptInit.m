function [W,H,Kj] = corruptInit(y,cPerSrc,stft_win_len,snr)
%CORRUPTINIT get a perturbed initialisation of NMF parameters
%
% INPUTS
%
%   y            : [M x I x J]  separated J (I-channel) source signals
%   cPerSrc      : [1 x 1]      number of NMF components for each source
%   stft_win_len : [1 x 1]      length of the STFT analysis window
%   snr          : [1 x 1]      level of corruption of initial parameters
%                               (dB) the higher the snr the less corrupted
%                               are the returned parameters W,H
%
% OUTPUT
%
%   W       : [F x K]                 matrix of bases
%   H       : [K x L]                 activation coefficients
%   Kj      : {J x 1} x [cPerSrc x 1] indexing of (columns of W and rows of
%                                     H), i.e. Kj{j} tells that W(:,Kj{j})
%                                     and H(Kj{j},:) are the NMF parameters
%                                     atributed to source j.
%
% version 29 March 10:40 PM

% [J x M] chose one of the microphones to estimate the psd of mono-sources
s = permute( y(:,1,:) , [3 1 2] ); % y : [M x I x J]

% J : number of sources, M : number of time samples
[J,M] = size(s);

% Calculation of "noisy" initial sources for intialization of NMF
% parameters: each source is corrupted by the other sources at given snr
s_j = zeros(J,M);
for j = 1:J
    sel = (1:J);
    sel = sel(sel~=j);
    s_0 = sum(s(sel,:),1); % sum of other sources
    s_j(j,:) = s(j,:) + 10^(-snr/20)*std(s(j,:))/std(s_0(:))*s_0;
end;

% [F x L x J] STFT of corrupted monochannel sources
S_j = stft_multi(s_j,stft_win_len);

fprintf('Initializing parameters ..\n');

% [F x K], [K x L] factorize S_j via Kullback Leibler NMF
[W,H,Kj] = Init_KL_NMF_fr_sep_sources(S_j,cPerSrc);










