# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /cps/home/kbreczin/Courses/Parallel/FinalProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cps/home/kbreczin/Courses/Parallel/FinalProject/build

# Include any dependencies generated for this target.
include CMakeFiles/nbody.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nbody.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nbody.dir/flags.make

CMakeFiles/nbody.dir/src/nbody.cu.o: CMakeFiles/nbody.dir/flags.make
CMakeFiles/nbody.dir/src/nbody.cu.o: ../src/nbody.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cps/home/kbreczin/Courses/Parallel/FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/nbody.dir/src/nbody.cu.o"
	/usr/local/cuda-11.4/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /cps/home/kbreczin/Courses/Parallel/FinalProject/src/nbody.cu -o CMakeFiles/nbody.dir/src/nbody.cu.o

CMakeFiles/nbody.dir/src/nbody.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/nbody.dir/src/nbody.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/nbody.dir/src/nbody.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/nbody.dir/src/nbody.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/nbody.dir/src/nbody.cu.o.requires:

.PHONY : CMakeFiles/nbody.dir/src/nbody.cu.o.requires

CMakeFiles/nbody.dir/src/nbody.cu.o.provides: CMakeFiles/nbody.dir/src/nbody.cu.o.requires
	$(MAKE) -f CMakeFiles/nbody.dir/build.make CMakeFiles/nbody.dir/src/nbody.cu.o.provides.build
.PHONY : CMakeFiles/nbody.dir/src/nbody.cu.o.provides

CMakeFiles/nbody.dir/src/nbody.cu.o.provides.build: CMakeFiles/nbody.dir/src/nbody.cu.o


# Object files for target nbody
nbody_OBJECTS = \
"CMakeFiles/nbody.dir/src/nbody.cu.o"

# External object files for target nbody
nbody_EXTERNAL_OBJECTS =

CMakeFiles/nbody.dir/cmake_device_link.o: CMakeFiles/nbody.dir/src/nbody.cu.o
CMakeFiles/nbody.dir/cmake_device_link.o: CMakeFiles/nbody.dir/build.make
CMakeFiles/nbody.dir/cmake_device_link.o: CMakeFiles/nbody.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cps/home/kbreczin/Courses/Parallel/FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/nbody.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nbody.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nbody.dir/build: CMakeFiles/nbody.dir/cmake_device_link.o

.PHONY : CMakeFiles/nbody.dir/build

# Object files for target nbody
nbody_OBJECTS = \
"CMakeFiles/nbody.dir/src/nbody.cu.o"

# External object files for target nbody
nbody_EXTERNAL_OBJECTS =

nbody: CMakeFiles/nbody.dir/src/nbody.cu.o
nbody: CMakeFiles/nbody.dir/build.make
nbody: CMakeFiles/nbody.dir/cmake_device_link.o
nbody: CMakeFiles/nbody.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cps/home/kbreczin/Courses/Parallel/FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable nbody"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nbody.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nbody.dir/build: nbody

.PHONY : CMakeFiles/nbody.dir/build

CMakeFiles/nbody.dir/requires: CMakeFiles/nbody.dir/src/nbody.cu.o.requires

.PHONY : CMakeFiles/nbody.dir/requires

CMakeFiles/nbody.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nbody.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nbody.dir/clean

CMakeFiles/nbody.dir/depend:
	cd /cps/home/kbreczin/Courses/Parallel/FinalProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cps/home/kbreczin/Courses/Parallel/FinalProject /cps/home/kbreczin/Courses/Parallel/FinalProject /cps/home/kbreczin/Courses/Parallel/FinalProject/build /cps/home/kbreczin/Courses/Parallel/FinalProject/build /cps/home/kbreczin/Courses/Parallel/FinalProject/build/CMakeFiles/nbody.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nbody.dir/depend

