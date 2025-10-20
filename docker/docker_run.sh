#Start a container with code on host (default)
docker run -t -i --hostname=docker --user modeluser -v ~/models:/code -v ~/models/inputdata:/inputdata -v elmoutput:/output elmv3 /bin/bash

