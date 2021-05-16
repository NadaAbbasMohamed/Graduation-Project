% Verifying the existance of python readable version
pyversion

file_path = 'path_to_file.py';

% Show the python module path to verify its readablility
path_to_module = fileparts(which(file_path));

% If path is not interpreted, link it to the python directory path
if count(py.sys.path,path_to_module)==0
    insert(py.sys.path,int32(0),path_to_module)
end

system('python webcam_detect.py');
#cap, start_time = py.webcam_detect_functions.prepare_for_run()
#if cap == 0   
    #return
#end
#fps = 0;
#while(cap ~=0)
   #r_image,k,fps,cap = py.webcam_detect_functions.take_frame_and_process(cap, fps);
#end

#py.webcam_detect_functions.prepare_for_end(start_time, fps);

#py.webcam_detect_functions.release_cam(cap);
