# Campaing Planning Tool (CPT): Python library for planning and designing scanning lidar measurement campaigns

## Why CPT?
Planning scanning lidar measurement campaigns is not a trivial task. There are many constraints, originating  both from the campaign site as well from the lidar technology, which a campaign planner needs to consider to derive the best possible layout of the campaign. The same can said for configuring scanning lidars to acquire high-quality measurements, or simply measurements that make sense. 

These tasks have been typically done ad-hoc and manually, thus requiring lidar experts. However, since 2018 a work has been put to digitilize these process.

After almost a decade of planning and configuring scanning lidar measurement campaign, the accumulated experience and knowledge has been converted in the Campaign Planning Tool (short CPT) to simplify the above mentioned tasks, making the scanning lidar technology useful beyond a handful of experts. 

## What CPT is capable of doing?
The CPT provides users with 


A WindScanner system consisting of two synchronized scanning lidar potentially represents a cost-effective solution for multi-point measurements, especially in complex terrain. However, the system limitations and limitations imposed by the wind farm site are detrimental to the installation of scanning lidars and the number and location of the measurement positions. To simplify the process of finding suitable measurement positions and associated installation locations for the WindScanner system we have devised a campaign planning workflow. The workflow consists of four phases:

1. Based on a preliminary wind farm layout, we generate optimum measurement positions using a greedy algorithm and a measurement ’representative radius’;

2. Areas where the lidar cannnot or should not be installed (due to terrain constraints such as lakes or  line-of-sight  blockage due to the terrain) are excluded - this determines the possible positions for the first lidar;

3. Possible positions fro the second lidar are defined based on the constraints related to the angles between theliar beams to abotine relible measurments;

4.  A trajectory through the measurement positions is generated for each lidar beam by applying the traveling salesman problem (TSP).

The above-described workflow has been digitilized into the so-called Campaign Planning Tool (CPT) currently provided as a Python library which allows users an effective way to plan measurement campaigns with WindScanner systems.

The presentation about the tool from the WESC 2019 conference can be found here:<br>
https://zenodo.org/record/3247797#.XR37V6eQ3RY

Preliminary results of the CPT application can be found here:<br>
https://data.dtu.dk/collections/Campaign_Planning_Tool_results_for_three_sites_in_complex_terrain/4559624

# Installation - Simple Way

Step 1: install Anaconda

Step 2: in bash type: 
	
	conda config --append channels conda-forge

Step 3: in bash type:
	
	conda create --name cpt --file requirements_conda.txt

Step 4: activate new enviroment: source activate cpt

Step 5: test pip: 
	
	which pip 
it should return “/anaconda3/envs/cpt/bin/pip”

Step 6: install remaining packages using following command:
	
	pip install -r requirements_pip.txt

Step 7: test new enviroment

