# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sloan/code/struck

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sloan/code/struck/build

# Include any dependencies generated for this target.
include CMakeFiles/struck.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/struck.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/struck.dir/flags.make

CMakeFiles/struck.dir/src/Config.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Config.cpp.o: ../src/Config.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/Config.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Config.cpp.o -c /home/sloan/code/struck/src/Config.cpp

CMakeFiles/struck.dir/src/Config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Config.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/Config.cpp > CMakeFiles/struck.dir/src/Config.cpp.i

CMakeFiles/struck.dir/src/Config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Config.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/Config.cpp -o CMakeFiles/struck.dir/src/Config.cpp.s

CMakeFiles/struck.dir/src/Config.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/Config.cpp.o.requires

CMakeFiles/struck.dir/src/Config.cpp.o.provides: CMakeFiles/struck.dir/src/Config.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Config.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Config.cpp.o.provides

CMakeFiles/struck.dir/src/Config.cpp.o.provides.build: CMakeFiles/struck.dir/src/Config.cpp.o

CMakeFiles/struck.dir/src/GraphUtils.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/GraphUtils.cpp.o: ../src/GraphUtils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/GraphUtils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/GraphUtils.cpp.o -c /home/sloan/code/struck/src/GraphUtils.cpp

CMakeFiles/struck.dir/src/GraphUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/GraphUtils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/GraphUtils.cpp > CMakeFiles/struck.dir/src/GraphUtils.cpp.i

CMakeFiles/struck.dir/src/GraphUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/GraphUtils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/GraphUtils.cpp -o CMakeFiles/struck.dir/src/GraphUtils.cpp.s

CMakeFiles/struck.dir/src/GraphUtils.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/GraphUtils.cpp.o.requires

CMakeFiles/struck.dir/src/GraphUtils.cpp.o.provides: CMakeFiles/struck.dir/src/GraphUtils.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/GraphUtils.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/GraphUtils.cpp.o.provides

CMakeFiles/struck.dir/src/GraphUtils.cpp.o.provides.build: CMakeFiles/struck.dir/src/GraphUtils.cpp.o

CMakeFiles/struck.dir/src/ImageRep.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/ImageRep.cpp.o: ../src/ImageRep.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/ImageRep.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/ImageRep.cpp.o -c /home/sloan/code/struck/src/ImageRep.cpp

CMakeFiles/struck.dir/src/ImageRep.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/ImageRep.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/ImageRep.cpp > CMakeFiles/struck.dir/src/ImageRep.cpp.i

CMakeFiles/struck.dir/src/ImageRep.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/ImageRep.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/ImageRep.cpp -o CMakeFiles/struck.dir/src/ImageRep.cpp.s

CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires

CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides: CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides

CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides.build: CMakeFiles/struck.dir/src/ImageRep.cpp.o

CMakeFiles/struck.dir/src/LaRank.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/LaRank.cpp.o: ../src/LaRank.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/LaRank.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/LaRank.cpp.o -c /home/sloan/code/struck/src/LaRank.cpp

CMakeFiles/struck.dir/src/LaRank.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/LaRank.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/LaRank.cpp > CMakeFiles/struck.dir/src/LaRank.cpp.i

CMakeFiles/struck.dir/src/LaRank.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/LaRank.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/LaRank.cpp -o CMakeFiles/struck.dir/src/LaRank.cpp.s

CMakeFiles/struck.dir/src/LaRank.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/LaRank.cpp.o.requires

CMakeFiles/struck.dir/src/LaRank.cpp.o.provides: CMakeFiles/struck.dir/src/LaRank.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/LaRank.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/LaRank.cpp.o.provides

CMakeFiles/struck.dir/src/LaRank.cpp.o.provides.build: CMakeFiles/struck.dir/src/LaRank.cpp.o

CMakeFiles/struck.dir/src/Features.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Features.cpp.o: ../src/Features.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/Features.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Features.cpp.o -c /home/sloan/code/struck/src/Features.cpp

CMakeFiles/struck.dir/src/Features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Features.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/Features.cpp > CMakeFiles/struck.dir/src/Features.cpp.i

CMakeFiles/struck.dir/src/Features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Features.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/Features.cpp -o CMakeFiles/struck.dir/src/Features.cpp.s

CMakeFiles/struck.dir/src/Features.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/Features.cpp.o.requires

CMakeFiles/struck.dir/src/Features.cpp.o.provides: CMakeFiles/struck.dir/src/Features.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Features.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Features.cpp.o.provides

CMakeFiles/struck.dir/src/Features.cpp.o.provides.build: CMakeFiles/struck.dir/src/Features.cpp.o

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o: ../src/HistogramFeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o -c /home/sloan/code/struck/src/HistogramFeatures.cpp

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/HistogramFeatures.cpp > CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/HistogramFeatures.cpp -o CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o

CMakeFiles/struck.dir/src/Sampler.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Sampler.cpp.o: ../src/Sampler.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/Sampler.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Sampler.cpp.o -c /home/sloan/code/struck/src/Sampler.cpp

CMakeFiles/struck.dir/src/Sampler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Sampler.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/Sampler.cpp > CMakeFiles/struck.dir/src/Sampler.cpp.i

CMakeFiles/struck.dir/src/Sampler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Sampler.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/Sampler.cpp -o CMakeFiles/struck.dir/src/Sampler.cpp.s

CMakeFiles/struck.dir/src/Sampler.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/Sampler.cpp.o.requires

CMakeFiles/struck.dir/src/Sampler.cpp.o.provides: CMakeFiles/struck.dir/src/Sampler.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Sampler.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Sampler.cpp.o.provides

CMakeFiles/struck.dir/src/Sampler.cpp.o.provides.build: CMakeFiles/struck.dir/src/Sampler.cpp.o

CMakeFiles/struck.dir/src/RawFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/RawFeatures.cpp.o: ../src/RawFeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/RawFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/RawFeatures.cpp.o -c /home/sloan/code/struck/src/RawFeatures.cpp

CMakeFiles/struck.dir/src/RawFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/RawFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/RawFeatures.cpp > CMakeFiles/struck.dir/src/RawFeatures.cpp.i

CMakeFiles/struck.dir/src/RawFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/RawFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/RawFeatures.cpp -o CMakeFiles/struck.dir/src/RawFeatures.cpp.s

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/RawFeatures.cpp.o

CMakeFiles/struck.dir/src/HaarFeature.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HaarFeature.cpp.o: ../src/HaarFeature.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/HaarFeature.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HaarFeature.cpp.o -c /home/sloan/code/struck/src/HaarFeature.cpp

CMakeFiles/struck.dir/src/HaarFeature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HaarFeature.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/HaarFeature.cpp > CMakeFiles/struck.dir/src/HaarFeature.cpp.i

CMakeFiles/struck.dir/src/HaarFeature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HaarFeature.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/HaarFeature.cpp -o CMakeFiles/struck.dir/src/HaarFeature.cpp.s

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides: CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides.build: CMakeFiles/struck.dir/src/HaarFeature.cpp.o

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/MultiFeatures.cpp.o: ../src/MultiFeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/MultiFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/MultiFeatures.cpp.o -c /home/sloan/code/struck/src/MultiFeatures.cpp

CMakeFiles/struck.dir/src/MultiFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/MultiFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/MultiFeatures.cpp > CMakeFiles/struck.dir/src/MultiFeatures.cpp.i

CMakeFiles/struck.dir/src/MultiFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/MultiFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/MultiFeatures.cpp -o CMakeFiles/struck.dir/src/MultiFeatures.cpp.s

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HaarFeatures.cpp.o: ../src/HaarFeatures.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/HaarFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HaarFeatures.cpp.o -c /home/sloan/code/struck/src/HaarFeatures.cpp

CMakeFiles/struck.dir/src/HaarFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HaarFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/HaarFeatures.cpp > CMakeFiles/struck.dir/src/HaarFeatures.cpp.i

CMakeFiles/struck.dir/src/HaarFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HaarFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/HaarFeatures.cpp -o CMakeFiles/struck.dir/src/HaarFeatures.cpp.s

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o

CMakeFiles/struck.dir/src/Tracker.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Tracker.cpp.o: ../src/Tracker.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/Tracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Tracker.cpp.o -c /home/sloan/code/struck/src/Tracker.cpp

CMakeFiles/struck.dir/src/Tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Tracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/Tracker.cpp > CMakeFiles/struck.dir/src/Tracker.cpp.i

CMakeFiles/struck.dir/src/Tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Tracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/Tracker.cpp -o CMakeFiles/struck.dir/src/Tracker.cpp.s

CMakeFiles/struck.dir/src/Tracker.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/Tracker.cpp.o.requires

CMakeFiles/struck.dir/src/Tracker.cpp.o.provides: CMakeFiles/struck.dir/src/Tracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Tracker.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Tracker.cpp.o.provides

CMakeFiles/struck.dir/src/Tracker.cpp.o.provides.build: CMakeFiles/struck.dir/src/Tracker.cpp.o

CMakeFiles/struck.dir/src/main.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/main.cpp.o: ../src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sloan/code/struck/build/CMakeFiles $(CMAKE_PROGRESS_13)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/struck.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/main.cpp.o -c /home/sloan/code/struck/src/main.cpp

CMakeFiles/struck.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sloan/code/struck/src/main.cpp > CMakeFiles/struck.dir/src/main.cpp.i

CMakeFiles/struck.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sloan/code/struck/src/main.cpp -o CMakeFiles/struck.dir/src/main.cpp.s

CMakeFiles/struck.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/struck.dir/src/main.cpp.o.requires

CMakeFiles/struck.dir/src/main.cpp.o.provides: CMakeFiles/struck.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/main.cpp.o.provides

CMakeFiles/struck.dir/src/main.cpp.o.provides.build: CMakeFiles/struck.dir/src/main.cpp.o

# Object files for target struck
struck_OBJECTS = \
"CMakeFiles/struck.dir/src/Config.cpp.o" \
"CMakeFiles/struck.dir/src/GraphUtils.cpp.o" \
"CMakeFiles/struck.dir/src/ImageRep.cpp.o" \
"CMakeFiles/struck.dir/src/LaRank.cpp.o" \
"CMakeFiles/struck.dir/src/Features.cpp.o" \
"CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/Sampler.cpp.o" \
"CMakeFiles/struck.dir/src/RawFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/HaarFeature.cpp.o" \
"CMakeFiles/struck.dir/src/MultiFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/HaarFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/Tracker.cpp.o" \
"CMakeFiles/struck.dir/src/main.cpp.o"

# External object files for target struck
struck_EXTERNAL_OBJECTS =

../bin/struck: CMakeFiles/struck.dir/src/Config.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/GraphUtils.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/ImageRep.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/LaRank.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/Features.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/Sampler.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/RawFeatures.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/HaarFeature.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/Tracker.cpp.o
../bin/struck: CMakeFiles/struck.dir/src/main.cpp.o
../bin/struck: CMakeFiles/struck.dir/build.make
../bin/struck: CMakeFiles/struck.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/struck"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/struck.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/struck.dir/build: ../bin/struck
.PHONY : CMakeFiles/struck.dir/build

CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Config.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/GraphUtils.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/LaRank.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Features.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Sampler.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Tracker.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/main.cpp.o.requires
.PHONY : CMakeFiles/struck.dir/requires

CMakeFiles/struck.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/struck.dir/cmake_clean.cmake
.PHONY : CMakeFiles/struck.dir/clean

CMakeFiles/struck.dir/depend:
	cd /home/sloan/code/struck/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sloan/code/struck /home/sloan/code/struck /home/sloan/code/struck/build /home/sloan/code/struck/build /home/sloan/code/struck/build/CMakeFiles/struck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/struck.dir/depend

