# opencv3

Fast image segmentation using MaxFlow/MinCut Boykov-Kolmogorov algorithm

Opencv grabcut algorithim is pretty good, but slow for large images. One bottleneck is the  function
GCGraph::MaxFlow, which implements the max-flow/min-cut Boykov-Kolmogorov algorithm. 
We propose a simple parallel version of this algorithm, providing a significant speedup.

   - We extend the class GCGraph (cf. file gcgraph.hpp) with an overloaded function maxFlow(int r). It computes
   the max flow corresponding to a subgraph of the initial graph. 
   
   - We implement a function constructGCGraph_slim (cf. grabcut.cpp) building a partially reduced graph. The reduction is based 
    on a paper of Scheuermann and Rosenhahn : https://pdfs.semanticscholar.org/92df/9a469fe878f55cd0ef3d55477a5f787c47ba.pdf
    
   - We implement a mulithreaded version of the function estimateSegmentation() (cf. file grabcut.cpp), 
    using our overloaded function maxFlow. Threads run on disjoint subgraphs, corresponding to disjoint subregions of the image, thus       no synchronization is needed. The final max flow is the exact max flow : the segmentation is optimal, as with the sequential           algorithm
    
Tests with a 24 M pixel image :

     estimateSegmentation()                                      48 s           grabcut()                63 s
     
     parallel estimateSegmentation() (64 regions, 8 threads)      2,9 s         parallel grabcut()       18 s

 
 Superlinear speedup is achieved by constructing short paths first in maxFlow().
 
History

  Most recent branch is workers_no_quadtree
  
  Last version is a test-only version

files

 dist/sources/modules/imgproc/include/opencv2/imgproc.hpp
 
 dist/sources/modules/imgproc/src/gcgraph.hpp
 
 dist/sources/modules/imgproc/src/grabcut.cpp
 
 
 
