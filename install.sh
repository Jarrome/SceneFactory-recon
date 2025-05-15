# install our point rasterizer
cd fastpr
python setup.py install
cd ..
# install ext
python ext/__init__.py

# install DM-NPS
pip install -e .
