### Configure drivers
- Create a rule for our new device, replace OWNER and GROUP with your username and group:
```
echo "$(sed "s/OWNER=\"user\",GROUP=\"group\"/OWNER=\"\$USER\",GROUP=\"\$USER\"/g" <<'EOF'
SUBSYSTEM=="usb", ATTR{idProduct}=="0101", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0102", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0103", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0001", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0002", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0003", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0004", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0005", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="3580", ATTR{idVendor}=="0ac8", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="9331", ATTR{idVendor}=="05a3", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0300", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0301", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0302", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0303", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0304", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0305", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0306", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0307", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0201", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0202", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0203", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0204", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0205", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0206", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
SUBSYSTEM=="usb", ATTR{idProduct}=="0207", ATTR{idVendor}=="2dbb", MODE="0666",OWNER="user",GROUP="group"
EOF
)" | sudo tee /etc/udev/rules.d/55-iminect-usb.rules > /dev/null && sudo udevadm control --reload-rules && sudo udevadm trigger
```

- Install dependencies for SDK and library

```
sudo apt install libgl1-mesa-dev mesa-common-dev freeglut3-dev libglfw3-dev
```

- Add IMI SDK to the PATH

```
# Bash shell
export IMISDK_DIR="/home/$USER/A100\ SDK/Linux/ImiSDK-Linux-1.8.1"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IMISDK_DIR/lib

# Fish shell
set -gx IMISDK_DIR "/home/$USER/A100\ SDK/Linux/ImiSDK-Linux-1.8.1" 
set -gx LD_LIBRARY_PATH $LD_LIBRARY_PATH:$IMISDK_DIR/lib 

```

- Install required packages from enviroment.txt

