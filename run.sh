################################################################################
# Step 1 - Start the Mosca server
cd /home/workspace
cd webservice/server/node-server
node ./server.js &
################################################################################
# Step 2 - Start the GUI
cd /home/workspace
cd webservice/ui
npm run dev &
################################################################################
# Step 3 - FFmpeg Server
sudo ffserver -f ./ffmpeg/server.conf &

################################################################################
# Download Model pedestrian-detection-adas-0002
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
sudo ./downloader.py --name pedestrian-detection-adas-0002  -o /home/workspace
sudo ./downloader.py --name person-detection-retail-0013  -o /home/workspace
cd /home/workspace

# Step 4 - Run the code
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
################################################################################
# python main.py -i {path_to_input_file} -m {path_to_model} -l {path_to_cpu_extension} -d {device_type} -pt {probability_threshold}

# python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
#

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/pedestrian-detection-adas-0002/FP32/pedestrian-detection-adas-0002.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m  intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
