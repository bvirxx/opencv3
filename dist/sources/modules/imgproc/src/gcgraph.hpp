/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _CV_GCGRAPH_H_
#define _CV_GCGRAPH_H_

template <class TWeight> class GCGraph
{
public:
	TWeight sink_sigmaW, source_sigmaW; // holds sum of terminal weights for all nodes
    GCGraph();
    GCGraph( unsigned int vtxCount, unsigned int edgeCount );
    ~GCGraph();
    void create( unsigned int vtxCount, unsigned int edgeCount );
    int addVtx();
    void addEdges( int i, int j, TWeight w, TWeight revw );
    void addTermWeights( int i, TWeight sourceW, TWeight sinkW );
    TWeight maxFlow();
    bool inSourceSegment( int i );
	cv::Point getFirstP(const int i);
	void setFirstP(const int i, const cv::Point p);
	TWeight getSourceW(const int i);
	TWeight getSinkW(const int i);
	TWeight sumW( const int i );
	cv::Point edge(const int i, const int j);  // indices of edges (i,j) and (j,i), (-1,-1) if not found
	void removeEdge(const int i, const int j);
	void joinNodes(const int i, const int j);
	void addWeight(const int i, const int j, const TWeight w); // increase weight of edge (i,j) by w. Edge is created if needed.
	int searchSimpleEdges();
private:
    class Vtx
    {
    public:
        Vtx *next; // initialized and used in maxFlow() only
        int parent;
        int first;
        int ts;
        int dist;
        TWeight weight, sourceW;
        uchar t;
		cv::Point firstP; // list of pixels joined to node
    };
    class Edge
    {
    public:
        int dst;
        int next;
        TWeight weight;
    };

    std::vector<Vtx> vtcs;
    std::vector<Edge> edges;
    TWeight flow;
};

template <class TWeight>
GCGraph<TWeight>::GCGraph()
{
    flow = 0;
	sink_sigmaW = 0;
	source_sigmaW = 0;
}

template <class TWeight>
GCGraph<TWeight>::GCGraph( unsigned int vtxCount, unsigned int edgeCount )
{
    create( vtxCount, edgeCount );
}
template <class TWeight>
GCGraph<TWeight>::~GCGraph()
{
}
template <class TWeight>
void GCGraph<TWeight>::create( unsigned int vtxCount, unsigned int edgeCount )
{
    vtcs.reserve( vtxCount );
    edges.reserve( edgeCount + 2 );
    flow = 0;
}

template <class TWeight>
int GCGraph<TWeight>::addVtx()
{
    Vtx v;
    memset( &v, 0, sizeof(Vtx));
    vtcs.push_back(v);
	v.firstP = cv::Point(-1, -1); // init to empty list
    return (int)vtcs.size() - 1;
}

template <class TWeight>
void GCGraph<TWeight>::addEdges( int i, int j, TWeight w, TWeight revw )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );
    CV_Assert( j>=0 && j<(int)vtcs.size() );
    CV_Assert( w>=0 && revw>=0 );
    CV_Assert( i != j );

    if( !edges.size() )
        edges.resize( 2 );

    Edge fromI, toI;
    fromI.dst = j;
    fromI.next = vtcs[i].first;
    fromI.weight = w;
    vtcs[i].first = (int)edges.size();
    edges.push_back( fromI );

    toI.dst = i;
    toI.next = vtcs[j].first;
    toI.weight = revw;
    vtcs[j].first = (int)edges.size();
    edges.push_back( toI );
}

template <class TWeight>
cv::Point GCGraph<TWeight>::getFirstP(const int i)
{
	return vtcs[i].firstP;
}

template <class TWeight>
void GCGraph<TWeight>::setFirstP(const int i, const cv::Point p)
{
	vtcs[i].firstP = p;
}

template <class TWeight>
TWeight  GCGraph<TWeight>::getSourceW(const int i)
{
	return vtcs[i].sourceW;
}

template <class TWeight>
TWeight  GCGraph<TWeight>::getSinkW(const int i)
{
	return vtcs[i].sourceW - vtcs[i].weight;
}

// search for edge joining 2 vertices
template <class TWeight>
cv::Point  GCGraph<TWeight>::edge(const int i, const int j)
{
	int ind1 = -1, ind2 = -1;

	if (edges.size() == 0)
		return cv::Point(ind1, ind2);

	Edge e;
	int p;

	for (p = vtcs[i].first, e = edges[p]; p > 0; p= e.next, e=edges[p])
	{
		if (e.dst == j)
		{
			ind1 = p;
			break;
		}
	}
	for (p = vtcs[j].first, e = edges[p]; p > 0; p = e.next, e=edges[p])
	{
		if (e.dst == i)
		{
			ind2 = p;
			break;
		}
	}

	CV_Assert(abs(ind1 - ind2) <= 1);

	return cv::Point(ind1, ind2);
}

template <class TWeight>
void GCGraph<TWeight>::removeEdge(const int i, const int j)
{
	for (int p = vtcs[i].first, e = edges[p]; p > 0; p=e.next)
	{
		if (e.next == 0)
			if (e.dst == j)
			{
				vtcs[i].first = 0;
				e.dst = -1;
				e.next = 0;
				e.weight = 0;
				break;
			}
			else if (edges[e.next].dst == j)
			{
				e.next = edges[e.next].next;  // remove edge at index e.next
				edges[e.next].dst = -1;
				edges[e.next].next = 0;
				edges[e.next].weight = 0;

				break;
			}
	}

	for (int p = vtcs[j].first, e = edges[p]; p > 0; p = e.next)
	{
		if (e.next == 0)
			if (e.dst == i)
			{
				vtcs[j].first = 0;
				e.dst = -1;
				e.next = 0;
				e.weight = 0;
				break;
			}
			else if (edges[e.next].dst == i)
			{
				e.next = edges[e.next].next; // remove edge at index e.next
				edges[e.next].dst = -1;
				edges[e.next].next = 0;
				edges[e.next].weight = 0;
				break;
			}
	}
}

template <class TWeight>
void  GCGraph<TWeight>::joinNodes(const int i, const int j)
{

	removeEdge(i, j);

	for (int p = vtcs[i].first, e = edges[p]; p > 0; p = e.next)
	{
		cv::Point tmp = edge(j, e.dst);
		if (tmp == (-1, -1))
			addEdges(j, e.dst, e.weight, e.weight);  
		else
		{
			edges[tmp.x].weight += e.weight;
			edges[tmp.y].weight += e.weight;
		}
		removeEdge(i, e.dst);
	}
	addTermWeights(j, getSourceW(i), getSinkW(i));
	addTermWeights(i, -getSourceW(i), -getSinkW(i));

	CV_Assert( vtcs[i].first == 0 );
}

// sum of weights for all edges adjacent to vtcs[i], including source and sink
template <class TWeight>
TWeight GCGraph<TWeight>::sumW(const int i)
{
	TWeight s = 0;
	for (int p=vtcs[i].first; p > 0; )
	{
		Edge& e = edges[p];
		s += e.weight;
		p=e.next;
	}
	//return s + getSourceW(i) + getSinkW(i);
	return s + 2.0*vtcs[i].sourceW - vtcs[i].weight;
}

template <class TWeight>
void GCGraph<TWeight>::addWeight(const int i, const int j, const TWeight w)
{
	Point a = edge(i, j);
	
	if ( a == Point(-1, -1))
		addEdges(i, j, w, w);
	else
	{
		edges[a.x].weight += w;
		edges[a.y].weight += w;
	}	
}

template <class TWeight>
int GCGraph<TWeight>::searchSimpleEdges()
{
	if (edges.size() == 0)
		return 0;

	int count = 0;

	for (int i = 2; i < edges.size() - 1; i += 2)
	{
		double w = edges[i].weight;

		if ((w > 0.5 * sumW(edges[i].dst)) || (w > 0.5 * sumW(edges[i+ 1].dst)))
			count++;
	}

	for (int i = 0; i < vtcs.size(); i++)
	{
		double w = vtcs[i].sourceW;
		double s = 0.5*sumW(i);

		if ((w > 0.5*source_sigmaW) || (w > s))
			count++;

		w = vtcs[i].sourceW - vtcs[i].weight;
		if ((w > 0.5*sink_sigmaW) || (w > s))
			count++;
	}
	return count;
}

template <class TWeight>
void GCGraph<TWeight>::addTermWeights( int i, TWeight sourceW, TWeight sinkW )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );

	sink_sigmaW += sinkW;
	source_sigmaW += sourceW;

    TWeight dw = vtcs[i].weight;
    if( dw > 0 )
        sourceW += dw;
    else
        sinkW -= dw;
    flow += (sourceW < sinkW) ? sourceW : sinkW;
    vtcs[i].weight = sourceW - sinkW;
	vtcs[i].sourceW += sourceW;
}

template <class TWeight>
TWeight GCGraph<TWeight>::maxFlow()
{
    const int TERMINAL = -1, ORPHAN = -2;
    Vtx stub, *nilNode = &stub, *first = nilNode, *last = nilNode;
    int curr_ts = 0;
    stub.next = nilNode;
    Vtx *vtxPtr = &vtcs[0];
    Edge *edgePtr = &edges[0];

    std::vector<Vtx*> orphans;

    // initialize the active queue and the graph vertices
    for( int i = 0; i < (int)vtcs.size(); i++ )
    {
        Vtx* v = vtxPtr + i;
        v->ts = 0;
        if( v->weight != 0 )
        {
            last = last->next = v;
            v->dist = 1;
            v->parent = TERMINAL;
            v->t = v->weight < 0;
        }
        else
            v->parent = 0;
    }
    first = first->next;
    last->next = nilNode;
    nilNode->next = 0;

    // run the search-path -> augment-graph -> restore-trees loop
    for(;;)
    {
        Vtx* v, *u;
        int e0 = -1, ei = 0, ej = 0;
        TWeight minWeight, weight;
        uchar vt;

        // grow S & T search trees, find an edge connecting them
        while( first != nilNode )
        {
            v = first;
            if( v->parent )
            {
                vt = v->t;
                for( ei = v->first; ei != 0; ei = edgePtr[ei].next )
                {
                    if( edgePtr[ei^vt].weight == 0 )
                        continue;
                    u = vtxPtr+edgePtr[ei].dst;
                    if( !u->parent )
                    {
                        u->t = vt;
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                        if( !u->next )
                        {
                            u->next = nilNode;
                            last = last->next = u;
                        }
                        continue;
                    }

                    if( u->t != vt )
                    {
                        e0 = ei ^ vt;
                        break;
                    }

                    if( u->dist > v->dist+1 && u->ts <= v->ts )
                    {
                        // reassign the parent
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                    }
                }
                if( e0 > 0 )
                    break;
            }
            // exclude the vertex from the active list
            first = first->next;
            v->next = 0;
        }

        if( e0 <= 0 )
            break;

        // find the minimum edge weight along the path
        minWeight = edgePtr[e0].weight;
        CV_Assert( minWeight > 0 );
        // k = 1: source tree, k = 0: destination tree
        for( int k = 1; k >= 0; k-- )
        {
            for( v = vtxPtr+edgePtr[e0^k].dst;; v = vtxPtr+edgePtr[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                weight = edgePtr[ei^k].weight;
                minWeight = MIN(minWeight, weight);
                CV_Assert( minWeight > 0 );
            }
            weight = fabs(v->weight);
            minWeight = MIN(minWeight, weight);
            CV_Assert( minWeight > 0 );
        }

        // modify weights of the edges along the path and collect orphans
        edgePtr[e0].weight -= minWeight;
        edgePtr[e0^1].weight += minWeight;
        flow += minWeight;

        // k = 1: source tree, k = 0: destination tree
        for( int k = 1; k >= 0; k-- )
        {
            for( v = vtxPtr+edgePtr[e0^k].dst;; v = vtxPtr+edgePtr[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                edgePtr[ei^(k^1)].weight += minWeight;
                if( (edgePtr[ei^k].weight -= minWeight) == 0 )
                {
                    orphans.push_back(v);
                    v->parent = ORPHAN;
                }
            }

            v->weight = v->weight + minWeight*(1-k*2);
            if( v->weight == 0 )
            {
               orphans.push_back(v);
               v->parent = ORPHAN;
            }
        }

        // restore the search trees by finding new parents for the orphans
        curr_ts++;
        while( !orphans.empty() )
        {
            Vtx* v2 = orphans.back();
            orphans.pop_back();

            int d, minDist = INT_MAX;
            e0 = 0;
            vt = v2->t;

            for( ei = v2->first; ei != 0; ei = edgePtr[ei].next )
            {
                if( edgePtr[ei^(vt^1)].weight == 0 )
                    continue;
                u = vtxPtr+edgePtr[ei].dst;
                if( u->t != vt || u->parent == 0 )
                    continue;
                // compute the distance to the tree root
                for( d = 0;; )
                {
                    if( u->ts == curr_ts )
                    {
                        d += u->dist;
                        break;
                    }
                    ej = u->parent;
                    d++;
                    if( ej < 0 )
                    {
                        if( ej == ORPHAN )
                            d = INT_MAX-1;
                        else
                        {
                            u->ts = curr_ts;
                            u->dist = 1;
                        }
                        break;
                    }
                    u = vtxPtr+edgePtr[ej].dst;
                }

                // update the distance
                if( ++d < INT_MAX )
                {
                    if( d < minDist )
                    {
                        minDist = d;
                        e0 = ei;
                    }
                    for( u = vtxPtr+edgePtr[ei].dst; u->ts != curr_ts; u = vtxPtr+edgePtr[u->parent].dst )
                    {
                        u->ts = curr_ts;
                        u->dist = --d;
                    }
                }
            }

            if( (v2->parent = e0) > 0 )
            {
                v2->ts = curr_ts;
                v2->dist = minDist;
                continue;
            }

            /* no parent is found */
            v2->ts = 0;
            for( ei = v2->first; ei != 0; ei = edgePtr[ei].next )
            {
                u = vtxPtr+edgePtr[ei].dst;
                ej = u->parent;
                if( u->t != vt || !ej )
                    continue;
                if( edgePtr[ei^(vt^1)].weight && !u->next )
                {
                    u->next = nilNode;
                    last = last->next = u;
                }
                if( ej > 0 && vtxPtr+edgePtr[ej].dst == v2 )
                {
                    orphans.push_back(u);
                    u->parent = ORPHAN;
                }
            }
        }
    }
    return flow;
}

template <class TWeight>
bool GCGraph<TWeight>::inSourceSegment( int i )
{
    CV_Assert( i>=0 && i<(int)vtcs.size() );
    return vtcs[i].t == 0;
}

#endif
