## Project Overview
This project is part of CAT498 and focuses on applying artificial intelligence to catastrophe modeling. In catastrophe modeling, risk is usually analyzed through four major modules: hazard, exposure, vulnerability, and loss. For Milestone 1, this project only focuses on the hazard module. The case study used in this work is Hurricane Ian. The main objective of this milestone is to explore whether a Conditional Generative Adversarial Network (cGAN) can be used to learn and generate realistic hazard representations from hurricane-related input data. The long-term goal is to extend AI methods to the other catastrophe modeling modules in future milestones.

## Data

The project uses Hurricane Ian hazard-related data as the primary input for model development. The data includes geospatial and storm-related information used to represent the hazard intensity and spatial pattern of the event. This may include raster layers, track-based storm data, wind-related hazard information, and other processed inputs required for training the cGAN model.
At this stage, the data is being used only for the hazard module, meaning the project is focused on generating or learning hazard patterns rather than estimating exposure, physical damage, or financial loss. The data is preprocessed so that it can be used as input to the AI model and compared with target hazard outputs.

## Data Sources
Hurrine IAN Track : https://www.nhc.noaa.gov/gis/archive_besttrack_results.php?id=al09&name=Hurricane+IAN&year=2022&utm_source=chatgpt.com

Lee County Background : https://maps.leegov.com/datasets/f57536cdc5bf4ecb88618c5ec61a6305_0/explore?location=26.552848%2C-82.011629%2C10

## Goals

The goals of this Milestone 1 project are:

1. Understand the role of the hazard module in catastrophe modeling.

2. Prepare Hurricane Ian hazard-related data for AI-based modeling.

3. Implement a Conditional GAN (cGAN) for hazard generation.

4. Evaluate whether the generated hazard outputs resemble realistic hazard patterns.

5. Build a foundation for future work in the other catastrophe modeling modules.

## Future Work

This milestone is only the first step of the overall project. In future work, I plan to expand AI applications to the remaining catastrophe modeling modules:

Exposure module: represent assets and infrastructure at risk

Vulnerability module: estimate damage response to hazard intensity

Loss module: estimate financial impact and losses

This will help create a broader AI-integrated catastrophe modeling workflow.

## Tools and Methods

1. Python

2. Deep Learning

3. Conditional GAN (cGAN)

4. Geospatial / raster-based hazard data

5. Hurricane Ian case study

## Current Scope

This repository currently contains work related only to Milestone 1, which is limited to the hazard module. The project does not yet include full exposure, vulnerability, or loss modeling.
