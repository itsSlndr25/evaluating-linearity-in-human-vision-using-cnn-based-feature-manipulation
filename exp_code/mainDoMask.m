% DoSimon
%
% This is the main program for running a simultaneous masking 
% experiment 
%  
%	8/22/98		ccc 			Wrote it.
%  !! go to line 101 to change the exp condition
 
% Declare screen globals
clear;
close all;

global screenRectT  winT 
Screen('Preference', 'SkipSyncTests', 1)
% Control variables
logicTest = 0;
useMeasContrasts = 0;

redoFlag = 0;
nTrials = 40;

% LUTFile = 'LutTable_LutInfo_calibMon2_20240412User1';

% Prompt for user initials and experiment number
subInitString = input('Enter subject initials [ccc]: ','s');
if (isempty(subInitString))
	subInitString = 'ccc';
end
expNum  = input('Enter experiment number [1]: ');
if (isempty(expNum))
	expNum = 1;
end

% Construct condition and data filenames 
% based on experiment number and subject initials
expNumString = num2str(expNum);
condFileName = [subInitString expNumString '.con'];
dataFileName = [subInitString expNumString '.dat'];
% It is the file that stores the trial to trial information
trialFileName = [subInitString expNumString '.tri'];

% Read previous data for this subject and condition.
nCondsRun = 0;
textData = [];
dataFileID = fopen(dataFileName,'r');
if (dataFileID ~= -1)
	while (1)
		textDataLine = fgetl(dataFileID);
		if (~isstr(textDataLine))
			break;
		end
		nCondsRun = nCondsRun+1;
	end
	fclose(dataFileID);
end

% Open data file for appending.
dataFileID = fopen(dataFileName,'a');
if (dataFileID == -1)
	error(sprintf('Cannot open data file %s',dataFileName));
end

% Open condition file and read all of the******[:-4]*********/
% conditions already run to get them out
% of the way.
condFileID = fopen(condFileName,'r');
textCondLine = fgetl(condFileID);
for i=1:nCondsRun
	textCondLine = fgetl(condFileID);
	if (~isstr(textCondLine))
		error('All conditions are run');
	end
end

% Initialize hardware

ClockRandSeed;

% Open screens

[winT, screenRectT] = Screen('OpenWindow',1); 

% Start timer and open screens
t0 = GetSecs;
% Loop over experiments.  Each time through the
% we do one condition.  The subject is given the
% opportunity to continue or quit at the end

% claim the exp var
controlConDB = -1; % set = -1
initialConDB = -1000;
duration = 18;
ISI = 1200;
ITI = 350;

while (1)
	if (redoFlag == 1)
		redoFlag = 0;
	else
		% Read condition information
		textCondLine = fgetl(condFileID); 
		if (~isstr(textCondLine))
            Screen('CloseAll')
            fclose('all')
			error('All conditions are run');
        end
        
        % 
		rawCondLine = sscanf(textCondLine,'%f %f %f %f %f');
		conNum = rawCondLine(1);
        pcNo = rawCondLine(2);
        layerLabel = rawCondLine(3);
        referenceCond = rawCondLine(4);
        imgIndex = rawCondLine(5);
	end

	%Load Look up table file

	% Print out what we are doing this run
	fprintf('Condition number: %f\n',conNum);
	
	% Load masker and target files.  The file convention
	% is that the variable "quantizedImage" contains the
	% bitmap.  We rename it immediately for clarity.
	%
	% NOTE.  These variables may be very large and if there
	% are memory problems, we may need a method for writing them
	% into the frame buffer here and then clearing them.  This
	% would involve rethinking the relation between this main
	% program and the subroutine RunSimon.
  
    
    % Read the text file into a cell array
    fileID = fopen('source_img_order2.txt', 'r');
    s_files_order = textscan(fileID, '%s');
    fclose(fileID);
    s_files_order = s_files_order{:}; % cell to list
    
%     tempNum = randperm(length(s_files_order));
%     nImageNameOrder(i) = tempNum(1);
%     INO_temp = tempNum(1);
%     % Get the file name without extension
%     [~, sfile_name, ~] = fileparts(s_files_order{INO_temp});
    
    % testPar = 2.0:0.25:14.25
    
    % choose different scale for different condition
    if pcNo > 7
        targetContrast = linspace(log10(5.1), log10(15.0), 40);
    elseif pcNo > 6 ||(pcNo==1 && layerLabel==2)
        targetContrast = linspace(log10(3.1), log10(9.0), 40);
    else 
        targetContrast = linspace(log10(1.1), log10(5.0), 40);
    end
    
    step = targetContrast(2) - targetContrast(1);
    extension = targetContrast(end) + (1:40) * step;
    targetContrast = [targetContrast, extension];
    
	% Load color look-up tables.  Again, the variable names
	% in the files are generic and we rename appropriately
	% here.  If the measured contrast table exists, then
	% we use it rather than the nominal values.
	clear clutContrastsDB measContrastsDB
	
	%  Load Look up table file ( no need
% 	loadCommand = ['load CLUTS/' LUTFile];
% 	eval(loadCommand);
% 
%   	contrastDBT = realdB;
% 	imageClutsT = LutsMon;
% 	bgClutT = kron([1 1 1],bgClut);
% 	bgClutT(1,:)=[0 0 0];
% 	
% 	clear contrastDB LutsMon  bgClut  contrastDB realdB ;
	
	% Run the experiment.
	if (logicTest == 1)
		thresholdConDB = initialConDB;
	else
		[thresholdDB, slopeEst, nRedos, timeSecs, ...
		testCont, interval, responses] = ... # ImageNameOrder
		  mainRunMask(initialConDB,...
			duration, ISI, ITI, targetContrast, ...
			nTrials, s_files_order, pcNo, layerLabel, referenceCond, imgIndex);
	end
		
	% Update the data file unless there was an abort
	if (thresholdDB ~= -2000)
		newDataLine = sprintf('%s %s %s %s %s %s %s %s %s %s %s %s',...
			num2str(conNum),...
			num2str(pcNo),...
			num2str(controlConDB),num2str(thresholdDB),num2str(slopeEst),...
			num2str(nRedos),num2str(timeSecs),...
			num2str(duration),num2str(ISI),num2str(ITI), num2str(layerLabel), num2str(imgIndex));
		fprintf(dataFileID,'%s\n',newDataLine);
		fclose(dataFileID);
		dataFileID = fopen(dataFileName,'a');
		if (dataFileID == -1)
			error('Error reopening data file after write');
		end
		
		% Open trail-to-trial file for appending.
		trialFileID = fopen(trialFileName,'a');
		if (trialFileID == -1)
			error(sprintf('Cannot open data file %s',trialFileName));
		end

		fprintf(trialFileID,'%s\n',newDataLine);
		for i = 1: nTrials
				newDataLine = sprintf('%s %s %s %s %s %s',...
				num2str(i),...
				num2str(testCont(i)),...
				num2str(interval(i)),num2str(responses(i)),num2str(layerLabel), num2str(imgIndex));
				fprintf(trialFileID,'%s\n',newDataLine);
		end 
		fclose(trialFileID);
		
		%	Ask subject about doing more and process button response
		disp('Press abort to quit, any other button to continue')
		 [keyIsDown, secs, key, deltaSecs] = KbCheck();
        buttons = find(key==1);
		
		% Break out on abort.  Otherwise bump number of conditions run
		% and continue.
		if (buttons== 41)
			break;				
		else
			nCondsRun = nCondsRun+1;
		end
	else
		%	Ask subject about doing more and process button response
		disp('Press redo to repeat, any other button to quit')
		 [keyIsDown, secs, key, deltaSecs] = KbCheck();
          buttons=find(key==1);
		
		% On redo, we don't bump number of conditions run.
		% If the subject doesn't want to redo, he has to quit.
		if (buttons == 44)	% button 44 & 41 ?
			redoFlag = 1;
		else
			break;		
		end
	end	
end

% Final close of files
fclose(dataFileID);
fclose(condFileID);

% Stop timer
s1=GetSecs;

% Close screen
clear mex



