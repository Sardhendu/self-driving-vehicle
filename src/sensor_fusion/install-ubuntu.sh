#!/bin/bash
source /etc/os-release || echo 'Warning: /etc/os-release was not found'

if [[ " $ID_LIKE " == *' archlinux '* ]]; then
  pacman -S git libuv openssl gcc cmake make
else
  if [[ ! " $ID_LIKE " == *' debian '* ]]; then
    echo 'Warning: unidentified Linux distribution, assuming Debian-like'
  fi

  apt-get update
  apt-get install git libuv1-dev libssl-dev gcc g++ cmake make
fi

git clone https://github.com/uWebSockets/uWebSockets
cd uWebSockets
git checkout e94b6e1
mkdir build
cd build
cmake ..
make
make install
cd ../..
ln -s /usr/lib64/libuWS.so /usr/lib/libuWS.so
rm -r uWebSockets
