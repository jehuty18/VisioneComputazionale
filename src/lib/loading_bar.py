import sys

def counting_bar(i, n_files):
    sys.stdout.write("\r%d of %d" % (i, n_files))
    sys.stdout.flush()