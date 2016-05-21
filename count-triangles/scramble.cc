#include <iostream>
#include <string>
#include <cassert>
using namespace std;

bool check(string s1, string s2) {
	// check if string s1 is permutation of s2
	int count[256] = {0};

	for (int i = 0; i < s1.length(); i++)
		count[s1.at(i)]++;
	
	for (int i = 0; i < s2.length(); i++)
		count[s2.at(i)]--;

	for (int i = 0; i < 256; i++)
		if (count[i] != 0)
			return false;
	
	return true;
}

bool isScramble(string s1, string s2) 
{
	cout << "Trying : (" << s1 << ", " << s2 << ")" << endl;
	assert(s1.length() == s2.length());

	bool valid = check(s1, s2);

	if (!valid)
		return false;
	if (s1.length() == 1)
		return true; // valid

	// all ways of partition
	for (int i = 1; i < s1.length(); i++) {
		string s1a = s1.substr(0, i);
		string s1b = s1.substr(i);
		string s2a = s2.substr(0, i);
		string s2b = s2.substr(i);

		if (isScramble(s1a, s2a) && isScramble(s1b, s2b))
			return true;

		// forward partition didn't work. Try reverse.
		int num = s1.length() - i;
		string s3b = s2.substr(0, num);
		string s3a = s2.substr(num);
		if (isScramble(s1a, s3a) && isScramble(s1b, s3b))
			return true;
	}
	return false;
}

int main(int argc, char* argv[]) {
	string s1 = argv[1];
	string s2 = argv[2];
	// assume same size and permutations.
	assert(s1.length() == s2.length() && check(s1, s2));
	cout << "isScramble? : " << isScrambleAnkit(s1, s2) << endl; 
}
