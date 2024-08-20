function y = emd(x,W,H,Kj,maxNumIter,stft_win_len,fs)
%EMD  EM algorithm for joint source separation & Diarization 
%        
% INPUTS
%
%  x   : [I x M]             mixture signal with M samples at I microphones
%
%  W   : [F x K]             initial matrix of bases (of all K components)
%
%  H   : [K x L]             initial matrix of contributions (all K comp.)
%
%  Kj  : {J x 1} x [? x 1]   Kj is a cell-array with J elements (one for 
%                            each source). Let K = size(W,2) be the total
%                            number of components. Any element Kj{j}, j=1:J
%                            of Kj contains the indexes (of the columns of
%                            W and rows of H) that correspond to source j.
%                            For example Kj = { [1 2 3] , [4 5] , [6 7] }
%                            would set the number of sources to J = 3,
%                            and use W(:,[1 2 3]) * H([1 2 3],:) as the NMF
%                            for souce j=1, W(:,[4 5]) * H([4 5],:) as
%                            the NMF of source j=2, W(:,[6 7]) * H(:,[6 7])
%                            as the NMF for source j=3.
%
%  maxNumIter                number of EM iterations
%
%  stft_win_len              lenght of STFT analysis window in samples
%
%
% OUTPUTS
%
%  y  : [J x M x I]          estimates of the J source images (time domain)
%  
%  diarization.rttm          transcription of the activity of the sources
%                            in file output.rttm. Source#j in output.rttm 
%                            corresponds to y(j,:,:) signal estimate
%
% version 4 Augusut 2017, 15:30 PM




%    /\
%   /__\
%  /    \ R C H E T Y P E



%% A   Constants indexSets & functions

% [F x L x I] mix STFT
X = stft_multi( transpose(x), stft_win_len);   

% size
[F,L,I] = size(X); M = size(x,1); J = numel(Kj); N = pow2(J); 

% [J x N] encoding
E = floor( mod( bsxfun(@rdivide, 0 : N-1 , pow2(J-1:-1:0)' ) , 2 ) );

% [1 x 1 x N x J] convention is [F x L x N x J x J]
E = permute(E,[3 4 2 1]);

% [1 x 1 x N x J x J] outer products E(:,n) * E(:,n)' of clms of E
EE = bsxfun(@times,E,permute(E,[1:3 5 4]));

% [F x 1]
normX = sum( sum( X .* conj(X) , 2 ) , 3 );

% f(x) normalize a matrix by the sum of its columns
normalize = @(Z) bsxfun(@rdivide,Z,sum(Z,1));

% f(x) multiply [F x L x N x J x J] MATRIX A with [F x L x N x J] VECTOR b
zgemv4D = @(A,b) sum( bsxfun(@times , A , permute(b,[1:3 5 4]) ) ,5);
%%




%   _____
%     |
%     |
%   __|__  N I T I A L I S A T I O N



%% I   Initialisation
 
T = rand(N);    T = T/sum(T(:));    Z = ones(1,1,N)/N;   

% positive definitness for R update
eyeJ = 1e-7 * eye(J);

% [1 x 1] sensor noise variance
v = X(:)' * X(:) * 1e3 / numel(X);

% {F} x [I x J] initialise filters
A = cell(F,1);   A(:) = { ones(I,J) };

% {F} x [J x J] M-X creates it as cell, E-S|Z casts it on array 
U = cellfun(@(A) A'*A, A, 'uniformoutput', false);
%%






for iter = 1:maxNumIter
%   ____
%  |
%  |____
%  |
%  |____ - S|Z    S T E P



%% E-S|Z   Source inference given Z (source posterior GMM)

% UPDATE
%   p     : [F x L x 1 x J]     source prior precision
%   B     : [F x L x N x J]     E * A^H * X / v
%   U     : [F x 1 x N x J x J] filter square * EE / v
%   Vs    : [F x L x N x J x J] posterior covariance of S
%   detVs : [F x L x N]         determinants of Vs, computed as Vs is cell
%   S     : [F x L x N x J]     posterior mean of S (Vs times B)

% [F x 1 x 1 x J x J] cast U in array, cat yields a [J x J x F] divide by v
U = bsxfun(@rdivide, permute( cat(3,U{:}) , [3 4 5 1 2] ) , v );

% [F x 1 x N x J x J] multiply with all possible activities
U = bsxfun(@times,EE,U);

% {J x 1} x [F x L]
p = cellfun(@(Kj) W(:,Kj) * H(Kj,:), Kj, 'uniformoutput', false);

% [F x L x 1 x J] dim 4 = J
p = 1./cat(4,p{:});

% Vs coviarnace & determinats

% [F x L x 1 x J x J] allocate
R = zeros(F,L,1,J,J);

% [F x L x 1 x J x J] set p on R's diagonal via indexing
R(:,:,:,1:J+1:J*J) = p;

% [J x J x F x L x N] R = diag(p) + U
R = permute( bsxfun(@plus,R,U) , [4 5 1:3] );

% {F x L x N} x [J x J] cast in cell
Vs = cell(F,L,N);    for ind = 1:F*L*N, Vs{ind} = R(:,:,ind); end

% {F x L x N} x [J x J] posterior covariance of S
Vs = cellfun(@inv,Vs,'uniformoutput',false);

% [F x L x N] calculate det() before recasting Vs in an array
detVs = cellfun(@det,Vs);

% [F x L x N x J x J] cast in array, on reshape leading J is columns
Vs = permute( reshape( cat(1,Vs{:}) , J,F,L,N,J) , [2:4 1 5]);

% calculate S

% [F x 1 x I x J] actually this is A^H
B = conj( permute( cat(3,A{:}) , [3 4 1 2]) );

% [F x L x 1 x J] A^H * X/v
B = sum( bsxfun(@times,B,bsxfun(@rdivide,X,v)) ,3);

% [F x L x N x J] multiply with activities
B = bsxfun(@times,B,E);

% [F x L x N x J] source mean (at GMM), extend B in [F x L x N x 1 x J]
S = zgemv4D(Vs,B);
%%




%   ____
%  |
%  |____
%  |
%  |____ - Z    S T E P



%% E-Z   Diarization inference

% UPDATE
%   Z    : [1 x L x N]       marginal posterior of Z
%   iZ   : {1 x L} x [N x 1] emission probability of Z
%   fZ   : {1 x L} x [N x 1] forward  probability of Z
%   bZ   : {1 x L} x [N x 1] backward probability of Z

% [1 x L x N] log|Vs| + B'*S, eliminate imaginary, sum on F
r = sum(real( log(detVs) + sum(conj(B).*S,4) ),1);

% [N x L] subtract maximum over N, roll N on dim-1
r = permute( bsxfun(@minus,r,max(r,[],3)) , [3 2 1] );

% [N x L] instead of dividing by sum(exp(r)) we substract log(sum(exp(r)))
r = exp(bsxfun(@minus,r,log(sum(exp(r)))));

% {L} x [N x 1] cast in cell, neat M-Z step
iZ = cell(L,1);    for l = 1:L, iZ{l} = r(:,l); end

% {L} x [N x 1] initialise forward, use previous/initial Z(1)
fZ = [ { normalize(reshape(Z(1,1,:),[],1) .* iZ{1}) } ;  cell(L-1,1)];

% forward pass

for l=2:L
    % {L} x [N x 1] update forward prob.
    fZ(l) = cellfun(@(iZ,fZ) normalize(iZ .* (T * fZ)), iZ(l),fZ(l-1), 'uniformoutput', false);
end

% {L} x [N x 1] initialise backward
bZ = [cell(L-1,1) ; fZ(L)];

% backward pass

for l=L-1:-1:1
    % {L} x [N x 1] update backward prob.
    bZ(l) = cellfun(@(iZ,bZ) normalize(T' * (iZ .* bZ)), iZ(l),bZ(l+1), 'uniformoutput', false);
end

% marginal of Z

% {L} x [N x 1] marginal posterior prob. of Z
Z = cellfun(@(fZ,bZ) normalize(fZ .* bZ), fZ, bZ, 'uniformoutput', false);

% [1 x L x N] cast in array
Z = permute( cat(2,Z{:}) , [3 2 1] );
%%




%  |\  /|
%  | \/ |   S T E P



%% NMF learning, Filters, Noise

% [F x L x N x J] imaginary deflations
r = real(Vs(:,:,:,1:J+1:J*J)) + S .* conj(S);

% [F x L x 1 x J]
r = sum(bsxfun(@times,r,Z),3);

% NMF solve
for j=1:J
   [W(:,Kj{j}), H(Kj{j},:)]   = nmf_is( r(:,:,j) , 1, W(:,Kj{j})  ,  H(Kj{j},:) );
end

% calculate Y

% [F x L x 1 x J] src Img
Y = sum( bsxfun(@times, E , bsxfun(@times,Z,S) ) ,3 );

% linear

% [F x 1 x I x J] outer products XY^H sum on L
B = sum( bsxfun(@times,X,conj(Y)) ,2);

% [I x J x F] convenient dims
B = permute(B,[3 4 1 2]);

% {F} x [I x J] cast in cell
r = cell(F,1); for f = 1:F, r{f} = B(:,:,f); end

% quadratic

% [F x L x N x J x J] Vs + S*S^H, conj(S) permutes in [F x L x N x 1 x J]
B = Vs + bsxfun(@times,S,permute(conj(S),[1:3 5 4]));

% [F x 1 x N x J x J] sum L prior to multiply with EE
B = sum(bsxfun(@times,B,Z),2);

% [J x J x F] multiply with EE, sum over N, make dim-{1,2} leading 
B = permute( sum(bsxfun(@times,EE,B),3) , [4 5 1:3] );

% {F} x [J x J] cast in cell, offset
R = cell(F,1); for f = 1:F, R{f} = B(:,:,f) + eyeJ; end

% solve LS

% {F} x [I x J] r * inv(R), filter estimate
A = cellfun(@mrdivide, r, R, 'uniformoutput', false);

% {F} x [J x J] filter square (recast in array in E-S|Z )
U = cellfun(@(A) A'*A, A, 'uniformoutput', false);

% calculate v

% [F x 1] residuals, sum on L
v = cellfun(@(A,r,U,R) -2*A(:)'*r(:) + U(:)'*R(:), A,r,U,R);

% [F x 1] add ||x||^2_2 and normalize
v = real(normX + v) / (L*I) + 1e-7;
%%




%  |\  /|
%  | \/ | - Z    S T E P



%% M-Z transition matrix

% {1 x L-1} x [N x N] joint probability p( Z_{l} , Z_{l-1} )
r = cellfun(@(bZ,iZ,fZ) bZ .* iZ*fZ' .* T + 1e-7,  bZ(2:L), iZ(2:L), fZ(1:L-1), 'uniformoutput', false);

% {1 x L-1} x [N x N]
r = cellfun(@(r) r/sum(r(:)) , r ,'uniformoutput',false);

% [N x N] transition
T = normalize( sum(cat(3,r{:}),3) );



fprintf('pass: %d\n',iter);
end





%% estimates of MASS and diarization

% [F x L x I x J], A is {F}x{IxJ}                        Y is [FxLx1xJ] 
Y = bsxfun(@times, permute(cat(3,A{:}),[3 4 1 2]) , Y );

% [I x M x J] estimates of source images (time-domain)
y = zeros(M,I,J);

for j=1:J
    y(:,:,j) = transpose( istft_multi( Y(:,:,:,j) , M));
end

% [N x L] states must be in rows -> see HMMVITERBI @z2rttm
Z = permute(Z,[3 2 1]);

% [J x N] is 1x1xNxJ
E = permute(E,[4 3 1 2]);

% write diarization output in file
z2rttm('diarization.rttm',Z,T,E,M,fs,stft_win_len)