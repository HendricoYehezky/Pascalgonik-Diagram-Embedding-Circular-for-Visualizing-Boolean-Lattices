Pascalgonic Diagram: Circular Embedding for Visualizing Boolean Lattices

This repository contains the Python implementation of the Pascalgonic diagram, a circular embedding method for visualizing Boolean lattices.
The method arranges subsets of a Boolean lattice into concentric polygonal layers and assigns them to angular partitions using an assignment optimization approach based on the Hungarian algorithm.

The Pascalgonic construction aims to produce a clear geometric representation of the Boolean lattice structure while preserving relational proximity between subsets.

Repository Contents
1. Pascalgonic Adjacent-Layer Compactness Visualization
   This program generates an explanatory visualization illustrating the geometric relationship between adjacent layers of the Pascalgonic diagram.
   The figure highlights:
   Angular directions of subsets
   Partition sector centers
   Layer structure R2
   Compactness relationships between adjacent layers

   This visualization is primarily intended for illustrative and explanatory purposes, particularly for use in academic articles discussing the Pascalgonic construction.
   Main features:
   Geometric explanation of layer compactness
   Visualization of angular partitioning
   Illustration of subset directions


2. Pascalgonic Diagram Venn Program
   This program generates the complete Pascalgonic diagram for Boolean lattices.
   The implementation constructs a circular embedding of the Boolean lattice Bp using:
   Polygonal layer representation
   Angular partition sectors
   Hungarian assignment optimization for subset placement
   Each layer of the Boolean lattice is mapped to a ring of a regular polygon, where subsets are assigned to angular sectors according to a cost-minimization procedure.
   Main features:
   Construction of Pascalgonic diagrams for prime values of 𝑝
   Layer-by-layer optimization using the Hungarian algorithm
   Visualization of subset relations
   Highlighting of subsets related to a selected target subset
   Step-by-step Hungarian optimization tables
   Interactive interface for parameter selection
   Supported parameters include:
   prime dimension p
   target subset angular rotation
   cost parameters for assignment optimization

   Method Overview
   The Pascalgonic diagram is constructed through the following steps:
   Generate all subsets of a Boolean lattice Bp.
   Group subsets into layers according to their cardinality.
   Represent each layer as a ring of a regular polygon.
   Define angular partitions corresponding to sector centers.
   Construct a placement cost matrix based on:
   angular distance between subset direction and sector center
   displacement from initial ordering
   Solve the assignment problem using the Hungarian algorithm.
   Place subsets in their optimal angular sectors.
   Render the Pascalgonic diagram.
   This process produces a structured visualization of Boolean lattices that emphasizes relational compactness around a selected target subset.

3. Requirements
   The programs are implemented in Python and require the following libraries:
   a. numpy
   b. pandas
   c. scipy
   d. matplotlib
   e. ipywidgets


These dependencies are commonly available in Jupyter Notebook or Google Colab environments.

