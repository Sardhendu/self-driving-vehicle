



1. Install Term-3 Simulator
2. Convert it into binary
   chmod +x /location_of_unzip_file/term2_sim.app/Contents/MacOS/term2_sim_mac
3. Install the package with uWebHook
  -> chmod u+x install-mac.sh
  -> ./install-mac.sh
4. Build the Project
  -> chmod u+x ./build.sh
  -> ./build.sh

   When you see this error:
   ld: cannot link directly with dylib/framework, your binary is not an allowed client of /usr/lib/libcrypto.dylib for architecture x86_64

   1. Check files in your Mac with similar names
   	ls -lh lib{crypto,ssl}*
   	actual file provided = -rwxr-xr-x  1 root  wheel    32K Apr  6 15:46 libcrypto.dylib
   2. Check what provided by openssl
       1. brew list openssl | grep lib
       2. Openssl provided file = /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libcrypto.1.1.dylib
   3. The problem with new versions of MAC is that we don’t have the privilege t change anything in the lib folder, such as mv libssl.dylib libssl.dylib.bak error: mv: rename libssl.dylib to libssl.dylib.bak: Read-only file system
   4. mount /dev/disk1s5 on / (apfs, local, read-only, journaled). Says that the mount is in read-only model
   5. we remount in write mode
   6. So we remount in write mode
       1. Sudo mount -r -w /
  7. Now we rename the existing libssl.dylib and libcrypto.dylib
    -> sudo mv libssl.dylib libssl.dylib.bak
    -> sudo mv libcrypto.dylib libcrypto.dylib.bak
  8. Check if the rename was successful
    -> ls -lh lib{crypto,ssl}*
  9. Copy the libssl and libcrypto files from Openssl
    -> sudo cp /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libssl.1.1.dylib ./
    -> sudo cp /usr/local/Cellar/openssl@1.1/1.1.1g/lib/libcrypto.1.1.dylib ./
  10. Create the link
    sudo ln -s libssl.1.1.dylib libssl.dylib
    sudo ln -s libcrypto.1.1.dylib libcrypto.dylib



  For compilers to find we may need to set:

  export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
  export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"


About the Dataset:

1. Highway Map Data (highway_map.csv):
   This file containes 5 columns each indicating the way points in the map

   1. x

2. Data from the Simulator:

    * car_x:              x position of the car in the map coordinate frame
    * car_y:              y position of the car in the map coordinate frame
    * car_s:              s position (longitudinal displacement) of the car in the frenet coordinate frame
    * car_d:              d position (lateral displacement) of the car in the frenet coordinate frame
    * car_yaw:            car's yaw angle in the map coordinate frame
    * car_speed:          speed of the car in miles/hr

    * previous_path_x:    list of x positions in map frame of the polynomial path in the previous (t-1) state.
    * previous_path_x:    list of y positions in map frame the polynomial path in the previous (t-1) state

    * sensor_fusion:      list(list)
