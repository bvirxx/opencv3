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
	TWeight sink_sigmaW, source_sigmaW; // sums of terminal weights for all nodes. Note source to sink weight is not added.
	TWeight stotW; // source to sink weight
	GCGraph();
	GCGraph(unsigned int vtxCount, unsigned int edgeCount);
	~GCGraph();
	void create(unsigned int vtxCount, unsigned int edgeCount);
	int addVtx();
	int addVtx(int r);
	void addEdges(int i, int j, TWeight w, TWeight revw);
	//void addWeight(const int i, const int j, const TWeight w); // increase weight of edge(i,j) by w. Edge is created if needed.
	void addTermWeights(int i, TWeight sourceW, TWeight sinkW);
	TWeight maxFlow();
	TWeight maxFlow(int r, int mask, double * result_ptr);
	inline bool inSourceSegment(int i);
	cv::Point getFirstP(const int i);
	void setFirstP(const int i, const cv::Point p);
	TWeight getSourceW(const int i);
	TWeight getSinkW(const int i);
	TWeight sumW(const int i);  // TODO not used; verify and remove
	inline int edge(const int i, const int j);  // index of edge (i,j), NOEDGE if not found. Use REVERSE(i) for reverse edge.
	// interface for graph reduction
	//void removeEdge(const int i, const int j);
	//void joinNodes(const int i, const int j);
	//void joinSink(const int i);
	//void joinSource(const int i);
	//cv::Point searchSimpleEdges(int startE, int startV, const bool first);
	//int reduce();
private:
	class Vtx
	{
	public:
		Vtx *next; // initialized and used in maxFlow() only
		int parent;
		int first;
		int ts;
		int dist;
		TWeight weight, sourceW, sumW;  // weight=sourceW-sink; sumW=sum of weights for all adjacent edges, including t-links
		uchar t;
		cv::Point firstP; // list of pixels joined to node
		int region;
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
	stotW = 0;
}

template <class TWeight>
GCGraph<TWeight>::GCGraph(unsigned int vtxCount, unsigned int edgeCount)
{
	create(vtxCount, edgeCount);
}
template <class TWeight>
GCGraph<TWeight>::~GCGraph()
{
}
template <class TWeight>
void GCGraph<TWeight>::create(unsigned int vtxCount, unsigned int edgeCount)
{
	vtcs.reserve(vtxCount);
	edges.reserve(edgeCount + 2);
	flow = 0;
}

template <class TWeight>
int GCGraph<TWeight>::addVtx()
{
	Vtx v;
	memset(&v, 0, sizeof(Vtx));
	v.firstP = cv::Point(-1, -1); // init to empty list
	vtcs.push_back(v);
	return (int)vtcs.size() - 1;
}

template <class TWeight>
int GCGraph<TWeight>::addVtx(int r)
{
	Vtx v;
	memset(&v, 0, sizeof(Vtx));
	v.firstP = cv::Point(-1, -1); // init to empty list
	v.region = r;
	vtcs.push_back(v);
	return (int)vtcs.size() - 1;
}

template <class TWeight>
void GCGraph<TWeight>::addEdges(int i, int j, TWeight w, TWeight revw)
{
	CV_Assert(i >= 0 && i<(int)vtcs.size());
	CV_Assert(j >= 0 && j<(int)vtcs.size());
	CV_Assert(w >= 0 && revw >= 0);
	CV_Assert(i != j);

	if (!edges.size())
		edges.resize(2);

	Edge fromI, toI;
	fromI.dst = j;
	fromI.next = vtcs[i].first;
	fromI.weight = w;
	vtcs[i].first = (int)edges.size();
	edges.push_back(fromI);
	vtcs[i].sumW += w;

	toI.dst = i;
	toI.next = vtcs[j].first;
	toI.weight = revw;
	vtcs[j].first = (int)edges.size();
	edges.push_back(toI);
	vtcs[j].sumW += revw;
}

# define NOEDGE -1;
#define REVERSE( p ) (( (( p )&0x01) == 0) ? ( p ) + 1 :  ( p ) - 1);  // index for reverse edge

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

// search for edges joining 2 vertices
template <class TWeight>
inline int GCGraph<TWeight>::edge(const int i, const int j)
{
	if (edges.size() == 0)
		return NOEDGE;

	for (int p = vtcs[i].first; p > 0;)
	{
		Edge& e = edges[p];
		if (e.dst == j)
			return p;
		p = e.next;
	}
	return NOEDGE;
}

//template <class TWeight>
//void GCGraph<TWeight>::removeEdge(const int i, const int j)
//{
//	for (int pred = -100, p = vtcs[i].first; p > 0;) //TODO use edge(i,j)
//	{
//		Edge& e = edges[p];
//		if (e.dst == j)
//		{
//			if (pred >=0)
//	            edges[pred].next = e.next;
//			else
//				vtcs[i].first = e.next;
//			e.dst = -1;
//			e.next = 0;
//			vtcs[i].sumW -= e.weight;
//			e.weight = 0;
//			break;
//		}
//		pred = p;
//		p = e.next;
//	}
//
//	for (int pred = -100, p = vtcs[j].first; p > 0;)
//	{
//		Edge& e = edges[p];
//		if (e.dst == i)
//		{
//			if (pred >= 0)
//				edges[pred].next = e.next;
//			else
//				vtcs[j].first = e.next;
//			e.dst = -1;
//			e.next = 0;
//			vtcs[j].sumW -= e.weight;
//			e.weight = 0;
//			break;
//		}
//		pred = p;
//		p = e.next;
//	}
//}

//template <class TWeight>
//void  GCGraph<TWeight>::joinNodes(const int i, const int j)
//{
//
//	removeEdge(i, j);
//
//	for (int p = vtcs[i].first;  p > 0; )
//	{
//		Edge& e = edges[p];   // TODO replace lines below by addWeight(j, e.dst, e.weight)
//
//		addWeight(j, e.dst, e.weight);  // TODO optimize
//
//		/*cv::Point tmp = edge(j, e.dst);
//		if (tmp == cv::Point(-1, -1))
//			addEdges(j, e.dst, e.weight, e.weight);  
//		else
//		{
//			edges[tmp.x].weight += e.weight;
//			edges[tmp.y].weight += e.weight;
//		}*/
//		p = e.next;
//		removeEdge(i, e.dst);  // TODO optimize : vtcs[i].first=0 and members of e set to remove-values
//		//p = e.next;
//	}
//	addTermWeights(j, getSourceW(i), getSinkW(i));
//	addTermWeights(i, -getSourceW(i), -getSinkW(i));
//
//	CV_Assert( vtcs[i].first == 0 );
//}

//template <class TWeight>
//void  GCGraph<TWeight>::joinSink(const int i)
//{
//	Vtx v = vtcs[i];
//	//connect all adjacent n-edges to sink
//	for (int p = v.first; p > 0;)
//	{
//		Edge& e = edges[p];
//		TWeight w = e.weight;
//		addTermWeights(e.dst, 0, w);  // TODO optimize
//		p = e.next;
//		removeEdge(i, e.dst);  // TODO optmize
//	}
//	// remove t-link to sink
//	sink_sigmaW -= v.sourceW - v.weight;
//	// update source to sink t-link
//	//s2tw += v.sourceW; // TODO uncomment when s2tw is a made a member field of GCGraph
//	// remove node
//	v.first = 0;
//	v.weight = 0;
//	v.sourceW = 0;   
//	v.sumW = 0;
//	//v.firstP=??
//}

//template <class TWeight>
//void  GCGraph<TWeight>::joinSource(const int i)
//{
//	Vtx v = vtcs[i];
//	//connect all adjacent n-edges to source
//	for (int p = v.first; p > 0;)
//	{
//		Edge& e = edges[p];
//		TWeight w = e.weight;
//		addTermWeights(e.dst, w, 0);  // TODO optimize
//		p = e.next;
//		removeEdge(i, e.dst);  // TODO optimize
//	}
//	// remove t-link to source
//	source_sigmaW -= v.sourceW;
//	// update sink to source t-link
//	//s2tw += v.sourceW - v.weight; // TODO uncomment when s2tw is a made a member field of GCGraph
//	// remove node
//	v.first = 0;
//	v.weight = 0;             // TODO encapsulate in a function jointoSource
//	v.sourceW = 0;
//	v.sumW = 0;
//	//v.firstP=??
//}

// sum of weights for all edges adjacent to vtcs[i], including source and sink
template <class TWeight>
inline TWeight GCGraph<TWeight>::sumW(const int i)
{                                                           // TODO not used; remove
	return vtcs[i].sumW;

	//TWeight s = 0;
	//for (int p=vtcs[i].first; p > 0; )
	//{
	//	Edge& e = edges[p];
	//	s += e.weight;
	//	p=e.next;
	//}
	////return s + getSourceW(i) + getSinkW(i);
	////if (!((s + 2.0*vtcs[i].sourceW - vtcs[i].weight) == vtcs[i].sumW))
	////	printf("sumW bad value %.2f", vtcs[i].sumW);

	////CV_Assert((s + 2.0*vtcs[i].sourceW - vtcs[i].weight) == vtcs[i].sumW);
	//return s + 2.0*vtcs[i].sourceW - vtcs[i].weight;
}

//template <class TWeight>
//void GCGraph<TWeight>::addWeight(const int i, const int j, const TWeight w)
//{
//	int a = edge(i, j);
//
//	if ( a == -1)
//		addEdges(i, j, w, w);
//	else
//	{
//		int b = REVERSE(a);
//		edges[a].weight += w;
//		edges[b].weight += w;
//		vtcs[i].sumW += w;
//		vtcs[j].sumW += w;
//	}	
//}

//template <class TWeight>
//cv::Point GCGraph<TWeight>::searchSimpleEdges(int startE, int startV, const bool first)
//{
//	if (edges.size() == 0)
//		return 0;
//
//	int count = 0;
//
//	startE = max(2, startE - startE % 2);
//
//	for (int i = startE; i < edges.size() - 1; i += 2)
//	{
//		double w = edges[i].weight;
//
//		if (w == 0)
//			continue;  // possibly removed edge. (dst is  -1) 
//
//		int d1 = edges[i].dst, d2=edges[i+1].dst;
//		if ((w > 0.5 * vtcs[d1].sumW) || (w > 0.5 * vtcs[d2].sumW))
//		{
//			count++;
//			if (first)
//				return cv::Point(0,i);
//		}
//	}
//
//	for (int i = startV; i < vtcs.size(); i++)
//	{   
//		if (vtcs[i].first == 0)  // no edges : removed node
//			continue;
//		double w = vtcs[i].sourceW;
//		double s = 0.5*vtcs[i].sumW;
//
//		if ((w > 0.5*source_sigmaW) || (w > s))
//		{
//			count++;
//			if (first)
//				return cv::Point (GC_JNT_FGD,i);
//		}
//
//		w = vtcs[i].sourceW - vtcs[i].weight;
//		if ((w > 0.5*sink_sigmaW) || (w > s))
//		{
//			count++;
//			if (first)
//				return cv::Point(GC_JNT_BGD,i);
//		}
//	}
//
//	//from the beginning
//	startE = min(startE, (int)edges.size() - 1);
//
//	for (int i = 2; i < startE; i += 2)
//	{
//		double w = edges[i].weight;
//
//		if (w == 0)
//			continue;  // possibly removed edge. (dst is  -1) 
//		
//		int d1 = edges[i].dst, d2 = edges[i + 1].dst;
//		if ((w > 0.5 * vtcs[d1].sumW) || (w > 0.5 * vtcs[d2].sumW))
//		{
//			count++;
//			if (first)
//				return cv::Point(0, i);
//		}
//	}
//
//	// from the beginning
//	startV = min(startV, (int)vtcs.size());
//
//	for (int i = 0; i < startV; i++)
//	{
//		if (vtcs[i].first == 0)  // no edges : removed node
//			continue;
//		double w = vtcs[i].sourceW;
//		double s = 0.5*vtcs[i].sumW;
//
//		if ((w > 0.5*source_sigmaW) || (w > s))
//		{
//			count++;
//			if (first)
//				return cv::Point(GC_JNT_FGD, i);
//		}
//
//		w = vtcs[i].sourceW - vtcs[i].weight;
//		if ((w > 0.5*sink_sigmaW) || (w > s))
//		{
//			count++;
//			if (first)
//				return cv::Point(GC_JNT_BGD, i);
//		}
//	}
//	if (!first)
//		printf("found %d simple edges\n",count);
//
//	return cv::Point(-10,-10); // BV_NO_VTX_FOUND;
//}

//template <class TWeight>
//int GCGraph<TWeight>::reduce()
//{
//	int count = 0;
//	int startE = 2;
//	int startV = 0;
//	for (;;)
//	{
//		Point ind = searchSimpleEdges(startE, startV,true);
//		if (ind.x >= 0)
//		{
//			joinNodes(edges[ind.y].dst, edges[ind.y + 1].dst);
//			count++;
//			startE = ind.y+2;
//		}
//		else if (ind.x == GC_JNT_FGD)
//		{
//			joinSource(ind.y);
//			//Vtx v = vtcs[ind.y];
//			//for (int p = v.first; p > 0;)
//			//{
//			//	Edge& e = edges[p];
//			//	TWeight w = e.weight;
//			//	addTermWeights(e.dst, w, 0);
//			//	p = e.next;//
//			//	removeEdge(ind.y, e.dst);
//			//}
//			//source_sigmaW -= v.sourceW;
//			//v.first = 0;
//			//v.weight = 0;             // TODO encapsulate in a function joinSource
//			//v.sourceW = 0;
//			////v.firstP=??
//			count++;
//			startV = ind.y + 1;
//			startE = edges.size();
//		}
//		else if (ind.x == GC_JNT_BGD)
//		{
//			joinSink(ind.y);
//			//Vtx v = vtcs[ind.y];
//			//for (int p = v.first; p > 0;)
//			//{
//			//	Edge& e = edges[p];
//			//	TWeight w = e.weight;
//			//	addTermWeights(e.dst, 0, w); //addTermWeights(e.dst, w, 0);
//			//	p = e.next;//
//			//	removeEdge(ind.y, e.dst);
//			//}
//			//sink_sigmaW -= v.sourceW - v.weight;
//			//v.first = 0;
//			//v.weight = 0;
//			//v.sourceW = 0;              // TODO encapsulate in a function jointoSink
//			////v.firstP=??
//			count++;
//			startV = ind.y + 1;
//			startE = edges.size();
//		}
//		else
//			break;
//	}
//	return count;
//}

template <class TWeight>
void GCGraph<TWeight>::addTermWeights(int i, TWeight sourceW, TWeight sinkW)
{
	CV_Assert(i >= 0 && i<(int)vtcs.size());

	sink_sigmaW += sinkW;
	source_sigmaW += sourceW;
	vtcs[i].sumW += (sourceW + sinkW);
	vtcs[i].sourceW += sourceW;


	TWeight dw = vtcs[i].weight;
	if (dw > 0)
		sourceW += dw;
	else
		sinkW -= dw;
	flow += (sourceW < sinkW) ? sourceW : sinkW;
	vtcs[i].weight = sourceW - sinkW;  // don't modify
	//vtcs[i].sourceW += sourceW;  // Wrong place
}

// Boykov-kolmogoroff algorithm
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

	int count = 0;
	// initialize the active queue and the graph vertices
	for (int i = 0; i < (int)vtcs.size(); i++)
	{
		Vtx* v = vtxPtr + i;
		v->ts = 0;
		if (v->weight != 0)
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
	for (;;)
	{
		//count++;
		Vtx* v, *u;
		int e0 = -1, ei = 0, ej = 0;
		TWeight minWeight, weight;
		uchar vt;

		// grow S & T search trees, find an edge connecting them
		while (first != nilNode)
		{
			v = first;
			if (v->parent)
			{
				vt = v->t;
				for (ei = v->first; ei != 0; ei = edgePtr[ei].next)
				{
					if (edgePtr[ei^vt].weight == 0)
						continue;
					u = vtxPtr + edgePtr[ei].dst;
					if (!u->parent)
					{
						u->t = vt;
						u->parent = ei ^ 1;
						u->ts = v->ts;
						u->dist = v->dist + 1;
						if (!u->next)
						{
							u->next = nilNode;
							last = last->next = u;
						}
						continue;
					}

					if (u->t != vt)
					{
						e0 = ei ^ vt;
						break;
					}

					if (u->dist > v->dist + 1 && u->ts <= v->ts)
					{
						// reassign the parent
						u->parent = ei ^ 1;
						u->ts = v->ts;
						u->dist = v->dist + 1;
					}
				}
				if (e0 > 0)
					break;
			}
			// exclude the vertex from the active list
			first = first->next;
			v->next = 0;
		}

		if (e0 <= 0)
			break;

		// find the minimum edge weight along the path
		minWeight = edgePtr[e0].weight;
		CV_Assert(minWeight > 0);
		// k = 1: source tree, k = 0: destination tree
		for (int k = 1; k >= 0; k--)
		{
			for (v = vtxPtr + edgePtr[e0^k].dst;; v = vtxPtr + edgePtr[ei].dst)
			{
				if ((ei = v->parent) < 0)
					break;
				weight = edgePtr[ei^k].weight;
				minWeight = MIN(minWeight, weight);
				CV_Assert(minWeight > 0);
			}
			weight = fabs(v->weight);
			minWeight = MIN(minWeight, weight);
			CV_Assert(minWeight > 0);
		}

		// modify weights of the edges along the path and collect orphans
		edgePtr[e0].weight -= minWeight;
		edgePtr[e0 ^ 1].weight += minWeight;
		flow += minWeight;

		// k = 1: source tree, k = 0: destination tree
		for (int k = 1; k >= 0; k--)
		{
			for (v = vtxPtr + edgePtr[e0^k].dst;; v = vtxPtr + edgePtr[ei].dst)
			{
				if ((ei = v->parent) < 0)
					break;
				edgePtr[ei ^ (k ^ 1)].weight += minWeight;
				if ((edgePtr[ei^k].weight -= minWeight) == 0)
				{
					orphans.push_back(v);
					v->parent = ORPHAN;
				}
			}

			v->weight = v->weight + minWeight*(1 - k * 2);
			if (v->weight == 0)
			{
				orphans.push_back(v);
				v->parent = ORPHAN;
			}
		}

		// restore the search trees by finding new parents for the orphans
		curr_ts++;
		while (!orphans.empty())
		{
			count++;
			Vtx* v2 = orphans.back();
			orphans.pop_back();

			int d, minDist = INT_MAX;
			e0 = 0;
			vt = v2->t;

			for (ei = v2->first; ei != 0; ei = edgePtr[ei].next)
			{
				if (edgePtr[ei ^ (vt ^ 1)].weight == 0)
					continue;
				u = vtxPtr + edgePtr[ei].dst;
				if (u->t != vt || u->parent == 0)
					continue;
				// compute the distance to the tree root
				for (d = 0;;)
				{
					if (u->ts == curr_ts)
					{
						d += u->dist;
						break;
					}
					ej = u->parent;
					d++;
					if (ej < 0)
					{
						if (ej == ORPHAN)
							d = INT_MAX - 1;
						else
						{
							u->ts = curr_ts;
							u->dist = 1;
						}
						break;
					}
					u = vtxPtr + edgePtr[ej].dst;
				}

				// update the distance
				if (++d < INT_MAX)
				{
					if (d < minDist)
					{
						minDist = d;
						e0 = ei;
					}
					for (u = vtxPtr + edgePtr[ei].dst; u->ts != curr_ts; u = vtxPtr + edgePtr[u->parent].dst)
					{
						u->ts = curr_ts;
						u->dist = --d;
					}
				}
			}

			if ((v2->parent = e0) > 0)
			{
				v2->ts = curr_ts;
				v2->dist = minDist;
				continue;
			}

			/* no parent is found */
			v2->ts = 0;
			for (ei = v2->first; ei != 0; ei = edgePtr[ei].next)
			{
				u = vtxPtr + edgePtr[ei].dst;
				ej = u->parent;
				if (u->t != vt || !ej)
					continue;
				if (edgePtr[ei ^ (vt ^ 1)].weight && !u->next)
				{
					u->next = nilNode;
					last = last->next = u;
				}
				if (ej > 0 && vtxPtr + edgePtr[ej].dst == v2)
				{
					orphans.push_back(u);
					u->parent = ORPHAN;
				}
			}
		}
	}
	printf("seq. maxFlow count %d\n", count);
	return flow;
}

// Boykov-kolmogoroff algorithm
template <class TWeight>
TWeight GCGraph<TWeight>::maxFlow(int r, int mask, double * result_ptr)
{
	const int TERMINAL = -1, ORPHAN = -2;
	Vtx stub, *nilNode = &stub, *first = nilNode, *last = nilNode;
	int curr_ts = 0;
	stub.next = nilNode;
	Vtx *vtxPtr = &vtcs[0];
	Edge *edgePtr = &edges[0];

	r = r&mask;

	clock_t tStart, tEnd;
	int count = 0;
	tStart = clock();

	std::vector<Vtx*> orphans;

	TWeight flow = 0;  // we must override graph.flow for concurrent writing

	// initialize the active queue and the graph vertices
	for (int i = 0; i < (int)vtcs.size(); i++)
	{
		Vtx* v = vtxPtr + i;
		if (((v->region)&mask )!= r)
			continue;
		v->ts = 0;
		if (v->weight != 0)
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
	for (;;)
	{
		//count++;
		Vtx* v, *u;
		int e0 = -1, ei = 0, ej = 0;
		TWeight minWeight, weight;
		uchar vt;

		// grow S & T search trees, find an edge connecting them
		while (first != nilNode)
		{
			v = first;
			if (v->parent)
			{
				vt = v->t;
				for (ei = v->first; ei != 0; ei = edgePtr[ei].next)
				{
					if (edgePtr[ei^vt].weight == 0)
						continue;
					if ((((vtxPtr + edgePtr[ei].dst)->region)&mask) != r)
						continue;
					u = vtxPtr + edgePtr[ei].dst;
					if (!u->parent)
					{
						u->t = vt;
						u->parent = ei ^ 1;
						u->ts = v->ts;
						u->dist = v->dist + 1;
						if (!u->next)
						{
							u->next = nilNode;
							last = last->next = u;
						}
						continue;
					}

					if (u->t != vt)
					{
						e0 = ei ^ vt;
						break;
					}

					if (u->dist > v->dist + 1 && u->ts <= v->ts)
					{
						// reassign the parent
						u->parent = ei ^ 1;
						u->ts = v->ts;
						u->dist = v->dist + 1;
					}
				}
				if (e0 > 0)
					break;
			}
			// exclude the vertex from the active list
			first = first->next;
			v->next = 0;
		}

		if (e0 <= 0)
			break;

		// find the minimum edge weight along the path
		minWeight = edgePtr[e0].weight;
		CV_Assert(minWeight > 0);
		// k = 1: source tree, k = 0: destination tree
		for (int k = 1; k >= 0; k--)
		{
			for (v = vtxPtr + edgePtr[e0^k].dst;; v = vtxPtr + edgePtr[ei].dst)
			{
				if ((ei = v->parent) < 0)
					break;
				weight = edgePtr[ei^k].weight;
				CV_Assert(((v->region)&mask) == r);   //TODO remove***********************************************
				minWeight = MIN(minWeight, weight);
				CV_Assert(minWeight > 0);
			}
			weight = fabs(v->weight);
			minWeight = MIN(minWeight, weight);
			CV_Assert(minWeight > 0);
		}

		// modify weights of the edges along the path and collect orphans
		edgePtr[e0].weight -= minWeight;
		edgePtr[e0 ^ 1].weight += minWeight;
		flow += minWeight;

		// k = 1: source tree, k = 0: destination tree
		for (int k = 1; k >= 0; k--)
		{
			for (v = vtxPtr + edgePtr[e0^k].dst;; v = vtxPtr + edgePtr[ei].dst)
			{
				if ((ei = v->parent) < 0)
					break;
				edgePtr[ei ^ (k ^ 1)].weight += minWeight;
				if ((edgePtr[ei^k].weight -= minWeight) == 0)
				{
					orphans.push_back(v);
					v->parent = ORPHAN;
				}
				CV_Assert(((v->region)&mask) == r);   //TODO remove***********************************************
			}

			v->weight = v->weight + minWeight*(1 - k * 2);
			if (v->weight == 0)
			{
				orphans.push_back(v);
				v->parent = ORPHAN;
			}
		}

		// restore the search trees by finding new parents for the orphans
		curr_ts++;
		while (!orphans.empty())
		{
			count++;
			Vtx* v2 = orphans.back();
			orphans.pop_back();

			int d, minDist = INT_MAX;
			e0 = 0;
			vt = v2->t;

			for (ei = v2->first; ei != 0; ei = edgePtr[ei].next)
			{
				if ((edgePtr[ei ^ (vt ^ 1)].weight == 0) || ((((vtxPtr + edgePtr[ei].dst)->region)&mask) != r))
					continue;
				u = vtxPtr + edgePtr[ei].dst;
				if (u->t != vt || u->parent == 0)
					continue;
				// compute the distance to the tree root
				for (d = 0;;)
				{
					if (u->ts == curr_ts)
					{
						d += u->dist;
						break;
					}
					ej = u->parent;
					d++;
					if (ej < 0)
					{
						if (ej == ORPHAN)
							d = INT_MAX - 1;
						else
						{
							u->ts = curr_ts;
							u->dist = 1;
						}
						break;
					}
					u = vtxPtr + edgePtr[ej].dst;
				}

				// update the distance
				if (++d < INT_MAX)
				{
					if (d < minDist)
					{
						minDist = d;
						e0 = ei;
					}
					for (u = vtxPtr + edgePtr[ei].dst; u->ts != curr_ts; u = vtxPtr + edgePtr[u->parent].dst)
					{
						u->ts = curr_ts;
						u->dist = --d;
					}
				}
			}

			if ((v2->parent = e0) > 0)
			{
				v2->ts = curr_ts;
				v2->dist = minDist;
				continue;
			}

			/* no parent is found */
			v2->ts = 0;
			for (ei = v2->first; ei != 0; ei = edgePtr[ei].next)
			{
				u = vtxPtr + edgePtr[ei].dst;
				if (((u->region)&mask) != r)
					continue;
				ej = u->parent;
				if (u->t != vt || !ej)
					continue;
				if (edgePtr[ei ^ (vt ^ 1)].weight && !u->next)
				{
					u->next = nilNode;
					last = last->next = u;
				}
				if (ej > 0 && vtxPtr + edgePtr[ej].dst == v2)
				{
					orphans.push_back(u);
					u->parent = ORPHAN;
					CV_Assert(((u->region)&mask) == r);   //TODO remove***********************************************
				}
			}
		}
	}
	*result_ptr = flow;
	tEnd = clock();

	//printf("region %d, mask %d, maxFlow time %.2f, iter %d\n", r, mask, (double)(tEnd - tStart), count);
	return flow;
}


template <class TWeight>
inline bool GCGraph<TWeight>::inSourceSegment(int i)
{
	CV_Assert(i >= 0 && i<(int)vtcs.size());
	return vtcs[i].t == 0;
}

#endif
