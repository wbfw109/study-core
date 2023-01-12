#!/bin/bash
git archive main --format=tar | xz -vz --threads=2 > ../study-core.tar.xz

# https://stackoverflow.com/questions/12075528/how-to-make-git-archive-in-7zip-format
# Note that xz and 7z use the same compression algorithm (LZMA). You can then unpack it with xz -d.
# .tar (tape archive) does not compress size, only archive.
# When I testing, .tar with xz will have most small size.
