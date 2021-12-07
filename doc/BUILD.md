## BUILD

This is a tutorial on how to build the engine on different platforms.

### Set up

```bash
git submodule update --init -recursive
```

### 1 MacOS

If you wanna help me to debug my CMakeLists.txt, just run below commands in the source directory.

```bash
cmake -B build
make -j4
```

**And the recommended way is to open the xcodeproj in the source directory.** Please add ffmpeg dynamic library to the project you want to build. Also you should build the libdatachannel.dylib and add to the project.

### 2 Linux

For linux version, you can directly go to this library I made, which is more light-weighted and controlled by cmake.

[Irrlicht with cmake on linux](https://github.com/MagicAtom/Irrlicht.git)

### 3 Windows

the code is verified to run on windows. But I do not have a windows laptop to test, so I didn't add the instructions here.
