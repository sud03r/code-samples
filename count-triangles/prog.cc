#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>

#define MAX 200000
using namespace std;

// counts number of common elements in 'sorted' vectors v1 and v2
int numCommonElements(const vector<int>& v1, int i1, 
					  const vector<int>& v2, int i2, 
					  vector<int>& commonElements)
{
	int rVal = 0;
	while (i1 < v1.size() && i2 < v2.size()) {
		if (v1[i1] == v2[i2]) {
			commonElements.push_back(v1[i1]);
			rVal++;
			i1++;
			i2++;
		} 
		else if (v1[i1] < v2[i2])
			i1++;
		else
			i2++;
	}
	return rVal;
}

class Graph {
	vector<int> nodeList[MAX];
public:
	void addEdge(int src, int dst) {
		if (src > dst) {
			int temp = src;
			src = dst;
			dst = src;
		}
		assert(src < MAX);
		nodeList[src].push_back(dst);
	}

	int sortEdgeList() {
		for (int i = 0; i < MAX; i++)
			sort(nodeList[i].begin(), nodeList[i].end());
	}

	int countTriangles() {
		int rVal = 0;
		for (int i = 0; i < MAX; i++) {
			const vector<int>& adj = nodeList[i];
			for (int j = 0; j < adj.size(); j++) {
				int curNodeId = adj[j];
				assert(curNodeId < MAX);
				const vector<int>& adjCurNode = nodeList[curNodeId];
				vector<int> commonElements;
				rVal += numCommonElements(adj, j+1, adjCurNode, 0, commonElements);
				for (int k = 0; k < commonElements.size(); k++)
					cout << i << " " << curNodeId << " " << commonElements[k] << endl;
			}
		}
		return rVal;
	}
};

int main(int arhc, char* argv[]) {
	Graph G;
	int src, dst;
	int start = clock();
	FILE* fp = fopen(argv[1], "r");
	if (!fp)
		cout << "Can't open" << argv[1] << endl;
	while(fscanf(fp, "%d %d", &src, &dst) > 0)
		G.addEdge(src, dst);
	fclose(fp);
	G.sortEdgeList();
	G.countTriangles();
	cout << "Program runtime " << (1.0 * (clock() - start) / CLOCKS_PER_SEC) << " seconds" << endl;
}
