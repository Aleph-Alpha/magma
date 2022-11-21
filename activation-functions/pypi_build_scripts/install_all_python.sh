# install python3.7 and python3.8
echo "Installing python3.7 and python3.8"
cwd=$(pwd)
yum install -y python3-devel.x86_64
yum install -y wget gcc openssl-devel bzip2-devel libffi-devel zlib-devel
yum -y groupinstall "Development Tools"
cd /usr/src
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
wget https://www.python.org/ftp/python/3.8.3/Python-3.8.3.tgz
wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz
tar xzf Python-3.7.9.tgz
tar xzf Python-3.8.3.tgz
tar xvf Python-3.9.5.tgz
cd /usr/src/Python-3.7.9
./configure --enable-optimizations
make altinstall
cd /usr/src/Python-3.8.3
./configure --enable-optimizations
make altinstall
cd /usr/src/Python-3.9.5
./configure --enable-optimizations
make altinstall
rm /usr/src/Python-3.7.9.tgz
rm /usr/src/Python-3.8.3.tgz
rm /usr/src/Python-3.9.5.tgz
cd $cwd
pip3 install auditwheel

echo "Installed python3.7 and python3.8"

#install the requirements for different pythons
echo "Installing all requirements for python3.6"
python3.6 -m pip install -U pip
python3.6 -m pip install wheel airspeed numpy torch
python3.6 -m pip install -r requirements.txt
echo "Installed all requirements for python3.6"
echo "Installing all requirements for python3.7"
python3.7 -m pip install -U pip
python3.7 -m pip install wheel airspeed numpy torch
python3.7 -m pip install -r requirements.txt
echo "Installed all requirements for python3.7"
echo "Installing all requirements for python3.8"
python3.8 -m pip install -U pip
python3.8 -m pip install wheel airspeed numpy torch
python3.8 -m pip install -r requirements.txt
echo "Installed all requirements for python3.8"
echo "Installing all requirements for python3.9"
python3.9 -m pip install -U pip
python3.9 -m pip install wheel airspeed numpy torch
python3.9 -m pip install -r requirements.txt
echo "Installed all requirements for python3.9"
