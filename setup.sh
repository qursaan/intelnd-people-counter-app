################################################################################
# Setup
sudo apt update
sudo apt install libzmq3-dev libkrb5-dev
# install Nodejs and its dependencies

# For MQTT/Mosca server:
cd /home/workspace
cd webservice/server
npm install
# For Web server:
cd ../ui
npm install

sudo npm install npm -g
rm -rf node_modules
npm cache clean
npm config set registry "http://registry.npmjs.org"
npm install
################################################################################
