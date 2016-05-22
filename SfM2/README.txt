
Instructions on how to run the program
========================================

* The main function is SfM2
	
	The format is
	SfM2 path-to-image1 path-to-image2 path-to-camera-instrinsics path-to-store-output-ply

	Examples:
	----------
	SfM2 ./quick-test/vase/1.JPG ./quick-test/vase/2.JPG ./quick-test/vase/intrinsic.txt ./quick-test/vase/vase.ply
	SfM2 ./quick-test/church/1.JPG ./quick-test/church/2.JPG ./quick-test/church/intrinsic.txt ./quick-test/church/church.ply 

* The first three lines of intrinsic.txt must form the 3x3 matrix K.

