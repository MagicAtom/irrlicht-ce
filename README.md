## Irrlicht-ce
We add a plugin to irrlicht game engine, to make it has the functions of cloud rendering. 

### 1 Build

For more details, please refer to the [BUILD.md](./doc/BUILD.md)

### 2 New Functions
- [x] Form video streaming locally  H.264
- [x] Stream with webrtc. 

### 3 New API Added

Below part is not finished yet. So some functions cannot receive parameters.

1. **recordScreen**

   If you want to record the game to a video,

   please use this api after every drawing,

   ```c++
   video->createEncoder();
   while(){
     smgr->drawAll();
   	video->recordScreen();
   }
   video->dropEncoder();
   ```

2. **publish**

   ```c++
   video->createStreamer();
   while(){
     smgr->drawAll();
     video->publish();
   }
   video->dropStreamer();
   ```
   
   

Thanks a lot for [Paul-Louis Ageneau](https://github.com/paullouisageneau), he helps us a lot about of webrtc.

