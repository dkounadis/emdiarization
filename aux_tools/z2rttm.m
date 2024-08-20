function [] = z2rttm(filename,Z,T,E,M,fs,stft_win_len)
% source activity from [N x L] matrix of posterior state probabilities

% N diarization states, L time frames
[N,L] = size(Z);

% [1 x L] optimal diarization state per frame (integer 1:N)
optSeq = hmmviterbi(1:L,T,Z);

% [N x L] cast to unary (Nx1 vector with single non-zero entry) 
eD = zeros(N,L); for l=1:L, eD(optSeq(l),l) = 1; end

% [J x L] source
Z = E*eD;

% smooth
Z = ~~medfilt1(Z,50,[],2);

J = size(Z,1);


fID = fopen(filename,'w');


for j=1:J
    
    % [stft_win_len x L]
    tmp = repmat(Z(j,:),stft_win_len,1);
    
    jump = stft_win_len/2;
    
    % [J x M] reconstructed binary signal, safety append
    z = zeros(M+stft_win_len,1);
    
    % if a frame is active
    for l=0:L-1
        z(l*jump+1:l*jump+stft_win_len,:) = or( z(l*jump+1:l*jump+stft_win_len,:) , tmp(:,l+1) );
    end
    
    % [1 x M] we look for a raise in the signal, i.e. for a 1,
    %         after we find a 1 we skip all following 1's until we find a 0
    %         then we look for another raise after this 0. 
    %         Assure that z(1)=0 and z(end)=0 so to stop the while
    z = z(1:M);   z(1) = 0;   z(end) = 0;
    
    % curr always tells a zero
    curr = 1;
    
    while curr < M
        
        % look for rise
        startM = find( z(curr:end) == 1, 1); % it may not find a rise
        
        if isempty(startM)
            
            curr = M; % no rise found, all signal is active
            
        else
            
            utterStart = curr + startM-1; % if empty
            
            % look for drop, skip the 0 at ftell
            durM = find( z(utterStart:end) == 0, 1);
            
            % .rttm
            fprintf(fID,'SPEAKER ID 1 %.2f %.2f <NA> <NA> estimatedSrc%d <NA>\n', utterStart/fs  ,  (durM-1)/fs ,  j);
            
            % equals M if end is reached
            curr = utterStart+durM;
            
        end 
    end
end