# This folder contains MATLAB scripts for running the psychophysical experiments used in the Vision Linearity project.

## 🏗️ Architecture

```
exp_code/
├── mainDoMask.m                # Main script for Experiment 1
├── mainRunMask.m               # Function to handle a single experimental run
├── mainDoAdd.m                 # Main script for Experiment 2
├── mainRunAdd.m                # Function to handle a single experimental run
├── mainPresentMonoStimulus.m   # Controls stimuli presentation within each trial
├── GetResponse.m               # Records participant responses
├── PSI/                        # Functions related to the adaptive Psi method
│   ├── ...
├── source_img_order.txt        # The list of source images (for ref condition)
```

## To start an experiment, simply run one of the two main scripts above.

### Main entry scripts
- `mainDoMask.m`
- `mainDoAdd.m`

These scripts:
- Ask for participant ID (3 letters, e.g., `cih`)
- Ask for experiment code (3 digits, e.g., `325`)
- Then will load condition file (`cih325.con`)
- Execute experimental runs

## Required Input Files

Before starting an experiment, prepare a condition file:
`[participantID][experimentCode].con`
Example: 
cih325.con # x2x.con indicates exp1
cih575.con # x7x.con indicates exp2

# ".con" data arrangement :
 - exp 1:
    index > pc index >  layer index > reference image condition > 
    original image index
 - exp 2:
    index > condition index >  layer index > reference image condition > original image index

## Output Files

After the experiment, two files will be generated:

- `[ID][Code].dat` → Data file for analysis  
- `[ID][Code].tri` → Trial log file (backup / detailed record)
Example: 
cih575.dat
cih575.tri
cgw625.dat
cih625.tri

## Notes

- Written in MATLAB(differ from analysis code) 
- Designed for controlled threshold estimate psychophysical experiments
- Uses adaptive threshold estimation via the Psi method