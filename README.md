# ID vs Video Verfication

This script works on verifying the provided ID against a live feed (cam) or recorded video.

## External Resources
***Download them in the same directory.***
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)


## Usage
```bash
$ python main.py path_to_ID path_to_video *Tolerance(optional)*
```
## Supported files
### Images 
All images are supported and the script convert them all to jpg.
### Videos
Only MP4 formate is tested for now.
