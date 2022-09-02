Coarse-to-Fine Registration with SE(3)-Equivariant Representations
==================================================================

Train
-----
    $ python train.py


Demo
----
    $ python demo.py --weights [checkpoints]


Installation
------------
    $ conda create -n cfreg python=3.8
    $ conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
    $ pip install -r requirement.txt
