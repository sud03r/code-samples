import sys
sys.path.insert(0, './src')
import driver

if len(sys.argv) < 2:
	print 'Usage: %s dataset-dir' % sys.argv[0]
else:
	dataset = sys.argv[1] + '/cifar-100-python/';
	driver.main(dataset)
