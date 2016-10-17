# opencv3 

Fast image segmentation using MaxFlow/MinCut Boykov-Kolmogorov algorithm 

We propose:

      - a parallel version of the function estimateSegmentation().
      
      - a modification of constructGCGraph() building a partially reduced graph. The modification is based 
        on a paper of Scheuermann and Rosenhahn : https://pdfs.semanticscholar.org/92df/9a469fe878f55cd0ef3d55477a5f787c47ba.pdf

Tests with a 24 M pixels image :

     estimateSegmentation()                       48 s
     parallel estimateSegmentation() (8 threads)   2,9 s
     
History

      Last version is a test-only version

files

     imgproc.hpp
     gcgraph.hpp
     grabcut.cpp
