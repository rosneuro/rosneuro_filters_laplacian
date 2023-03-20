# ROS-Neuro Laplacian filter

This ROS-Neuro filter plugin implements a Laplacian derivation filter.

## Algorithm:
The filter applies the Laplacian derivation to the data. For each provided sample and for each channel, it substracts the averaged value computed across the neighbour channels. Neighbours are identified with a cross policy:
```
         N3
         ||
 N1 == Channel == N2
         ||
         N4
```
It is **required** to provide the channel layout as parameter of the filter. The number of channels (*nchannels*) is optional: if it is not provided, the filter tries to deduce it from the layout (it considers the highest index as the number of channel. **Be carefull: this might be not true**). The channel layout is defined as a matrix with the index of the channels and 0 in the other positions:

## YAML configuration
```
LaplacianCfgTest:
  name: laplacian
  type: LaplacianWindowFloat
  params: 
    layout: " 0  0  1  0  0; 
              2  3  4  5  6;
              7  8  9 10 11; 
             12 13 14 15 16"
```
